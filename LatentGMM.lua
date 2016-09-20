require 'nn'
require 'GaussianCriterion'
require 'optim'
require 'nngraph'
require 'KLCriterion'
require 'Dirichlet'
require 'NormalGamma'
require 'Gaussian'
require 'Label'
local nninit = require 'nninit'




torch.manualSeed(1)
data = torch.load('save/spiral.t7')

local mnist = require 'mnist'
--data = mnist.traindataset().data:div(255):double()
--data = data:view(data:size(1), data:size(2)*data:size(3))


local N  = data:size(1)
local Dy = data:size(2)
local Dx = 2
local batch = 100
local batchScale = N/batch
local eta = 0.001
local eta_latent = 0.1
local optimiser = 'adam'
local latentOptimiser = 'sgd'
local max_Epoch = 10000
local K = 15
local max_iter = 500

-- ResNet 
function resNetBlock(inputSize, hiddenSize )
	local input = - nn.Identity()
	local resBranch  = input 
					  - nn.Linear(inputSize, hiddenSize):init('weight', nninit.normal, 0,0.001)
					  									:init('bias' , nninit.normal, 0, 0.001)
					  - nn.Tanh()
					  - nn.Linear(hiddenSize, inputSize):init('weight', nninit.normal, 0,0.001)
					  									:init('bias' , nninit.normal, 0, 0.001)
	local skipBranch = input 
					  - nn.Identity()
	local output 	 = {resBranch, skipBranch}
						- nn.CAddTable() 
	return nn.gModule({input}, {output})
end

function globalMixing()
	local phi = - nn.Identity() -- [K, N ]
	local hk  = - nn.Identity() -- [K, Dx]
	local Jk  = - nn.Identity() -- [K, Dx]

	local phiT = phi - nn.Transpose({1,2}) -- [N, K]
	local Ehk  = {phiT, hk} - nn.MM() -- [N, Dx]
	local EJk  = {phiT, Jk} - nn.MM() -- [N, Dx]

	return nn.gModule({phi, hk, Jk}, {Ehk, EJk})
end

function gaussainMeanfield()
	local hy  = - nn.Identity() -- [N, Dx]
	local Jy  = - nn.Identity() -- [N, Dx]
	local Ehk = - nn.Identity()
	local EJk = - nn.Identity()

	local hx = {hy, Ehk} - nn.CAddTable() -- mu/var
	local Jx = {Jy, EJk} - nn.CAddTable() -- -1/2(1/var)
 
	local var   = Jx - nn.MulConstant(-2)
				     - nn.Power(-1)
	local mean  = {hx, var} 
					 - nn.CMulTable()

	return nn.gModule({hy, Jy, Ehk, EJk}, {mean, var})
end

function createSampler()
	-- Sampler
	local mean   = - nn.Identity()
	local var    = - nn.Identity() 
	local rand   = - nn.Identity()
	local std    = var - nn.Power(0.5)
	local noise  = {std, rand}
				   - nn.CMulTable()
	local x 	 = {mean, noise}
				   - nn.CAddTable()			   

	return nn.gModule({mean, var, rand}, {x})
end

-- Network 
function createNetwork(Dy, Dx)
	local hiddenSize = 100
	-- Recogniser
	local input  = - nn.Identity()
	local hidden = input
				   - resNetBlock(Dy, hiddenSize)
				
	local mean     = hidden
				   - resNetBlock(Dy, hiddenSize)	

	local logVar   = hidden
					- nn.Linear(Dy, hiddenSize)
					- nn.Tanh()
					- nn.Linear(hiddenSize, Dy):init('bias' , nninit.normal, -5, 0.001)
					  						   

	local Jy 	   = logVar
					- nn.Exp()  -- Var
					- nn.Power(-1) -- 1/var
					- nn.MulConstant(-0.5) --  - 1/2Var
	local hy 	   = {mean, Jy}
					- nn.CMulTable() -- mean/(-2var)
					- nn.MulConstant(-2) -- mean/var

	local recogniser = nn.gModule( {input}, {hy, Jy})

	-- Generator
	local X_sample   = - nn.Identity()
	local h          = X_sample
					   - resNetBlock(Dy, hiddenSize)	

	local recon_mean =   h
					   - resNetBlock(Dy, hiddenSize)	
	local recon_logVar = h
						- nn.Linear(Dy, hiddenSize)
						- nn.Tanh()
						- nn.Linear(hiddenSize, Dy)

	local generator = nn.gModule({X_sample}, {recon_mean, recon_logVar})

	return recogniser, generator
end


-- Latent Model
local dir = nn.Dirichlet(K)
local NG = nn.NormalGamma(K,Dx)
local gaussian = nn.Gaussian(K, Dx, batch)
local label = nn.Label(K, batch)

-- set prior
local m0 = torch.Tensor(Dx):zero()
local l0 = torch.Tensor(Dx):fill(1)
local a0 = torch.Tensor(Dx):fill(1)
local b0 = torch.Tensor(Dx):fill(1)
NG:setPrior(m0, l0, a0, b0)
dir:setPrior(1000)

-- Initialise parameters
local m1 = torch.rand(K, Dx):add(-0.5):mul(2)
local l1 = torch.Tensor(K, Dx):fill(1)
local a1 = torch.Tensor(K, Dx):fill(10)
local b1 = torch.Tensor(K, Dx):fill(1)
local pi1 = torch.randn(K)
NG:setParameters(m1, l1, a1, b1)
dir:setParameters(pi1)

local latentContainer = nn.Container()
						:add(NG)
						:add(dir)
local latentParams, gradLatentParams = latentContainer:getParameters()


local recogniser, generator = createNetwork(Dy, Dx)
local sampler = createSampler()
local latentGMM = gaussainMeanfield()
local globalMixing = globalMixing()

local container = nn.Container()
				  :add(recogniser)
				  :add(generator)
				  

local ReconCrit = nn.GaussianCriterion( batchScale )
local KLCrit    = nn.KLCriterion(  batchScale )

local params, gradParams = container:getParameters()


function feval(param)

	if param ~= params then
        params:copy(param)
    end

	container:zeroGradParameters()
	latentContainer:zeroGradParameters()

	-- Recogniser
	local hy, Jy = unpack( recogniser:forward(y) )

	-- Get global expected stats
	local E_NG = NG:updateExpectedStats()
	local E_dir = dir:updateExpectedStats()
	local mean_x, var_x
	
	------ Latent GMM ----
	-- Initialise phi [KxN]
	local phi = label:reset()
	local Tx, Ehk, EJk
	local llh_prev = 0.0
	for i=1, max_iter do

		-- From {hy, Jy, phi, E_NG[1], E_NG[2]} -> {hx , Jx} -> {mean_x, var_x}-> {x, x2}
		Ehk, EJk	  = unpack( globalMixing:forward( {phi, E_NG[1], E_NG[2]} ) )
		mean_x, var_x = unpack( latentGMM:forward({hy, Jy, Ehk, EJk}) )
		
		Tx = {mean_x, mean_x:clone():pow(2):add(var_x) } --TODO fix this

		-- Get gaussian expected llh
		local llh = gaussian:getLogLikelihood(E_NG, Tx)

		-- Update Label parameters
		phi = label:setParameters(llh, E_dir)

		local sumllh = llh:sum()
		if torch.abs( llh_prev - llh:sum() ) < 1e-7 then 
			break
		end
		llh_prev = sumllh
		if i == max_iter then
			print('max iter reach')
		end

	end
	
	-- Compute Mixture stats
	local Txz = gaussian:getMixtureStats(phi, Tx, batchScale)

	-- Do sampling
	local rand  = torch.randn(var_x:size()):mul(1)
	local xs    = sampler:forward({mean_x, var_x, rand})
	
	-----------------------

	local recon     = generator:forward(xs)
	local reconLoss = ReconCrit:forward(recon, y)


	
	local gradRecon = ReconCrit:backward(recon, y)
	local gradXs    = generator:backward(xs, gradRecon)
	local gradMean, gradVar, __ = unpack( sampler:backward({mean_x, var_x, rand}, gradXs ) )
	local gradHy, gradJy, gradEhk, gradEJk = unpack( latentGMM:backward({hy, Jy, Ehk, EJk}, {gradMean, gradVar}) )
	recogniser:backward(y, {gradHy, gradJy})
	
	-- Update global parameters
	--local gradPHI, gradHK, gradJK = unpack( globalMixing:backward({phi, E_NG[1], E_NG[2]} ,{gradEhk, gradEJk}))
	--NG:backward(Txz, {gradHK, gradJK}) -- give NaN for sgd optimiser
	NG:backward(Txz)
	dir:backward(Txz)

	
	local var_k = EJk:clone():mul(-2):pow(-1)
	local mean_k = Ehk:clone():cmul(var_k)
	local KLLoss = KLCrit:forward({mean_x, var_x}, {mean_k, var_k})
	
	local gradMean, gradVar  = unpack( KLCrit:backward({mean_x, var_x}, {mean_k, var_k}) )
	local gradHy, gradJy, gradEhk, gradEJk = unpack( latentGMM:backward({hy, Jy, Ehk, EJk}, {gradMean, gradVar}) )
	recogniser:backward(y, {gradHy, gradJy})
	


	local loss   = reconLoss + KLLoss
	return loss, gradParams
end

function Lfeval(params)
	return __, gradLatentParams
end



for epoch = 1, max_Epoch do

	local indices = torch.randperm(N):long():split(batch)

	local recon = torch.Tensor():resizeAs(data):zero()
	local labels = torch.Tensor():resize(N):zero()
	local x_sample = torch.Tensor():resize(N, Dx):zero()
	local Loss = 0.0

	for t,v in ipairs(indices) do 
		xlua.progress(t, #indices)
		y = data:index(1,v)

		__, loss = optim[optimiser](feval, params, {learningRate = eta })
		__, __   = optim[latentOptimiser](Lfeval, latentParams, {learningRate = eta_latent})

		recon[{ { batch*(t-1) + 1, batch*t },{}}]    = generator.output[1]
		x_sample[{ { batch*(t-1) + 1, batch*t },{}}] = sampler.output
		labels[{{ batch*(t-1) + 1, batch*t }}]       = label:assignLabel()

		Loss = Loss + loss[1]
	end

	print("Epoch: " .. epoch .. " Loss: " .. Loss/N )
	torch.save('save/label.t7', labels)
	torch.save('save/recon.t7', recon)
	torch.save('save/xs.t7', x_sample)

	-- Latent plot --
	local m, l, a, b = unpack(NG:getBasicParameters())
	local cov = torch.Tensor(K, Dx, Dx)
	for k=1,K do
		cov[k] = torch.diag( torch.cdiv(b,a)[k] )
	end
	torch.save('save/m.t7', m)
	torch.save('save/cov.t7', cov)


end 











