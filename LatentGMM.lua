require 'nn'
require 'GaussianCriterion'
require 'optim'
require 'nngraph'
require 'KLCriterion'
require 'Dirichlet'
require 'NormalGamma'
require 'Gaussian'
require 'Label'

torch.manualSeed(1)
data = torch.load('save/spiral.t7')

local N  = data:size(1)
local Dy = data:size(2)
local Dx = 2
local batch = 1000
local batchScale = N/batch
local eta = 0.001
local eta_latent = 0.1
local optimiser = 'adam'
local latentOptimiser = 'sgd'
local max_Epoch = 10000
local K = 5
local max_iter = 500

-- Network 
function createNetwork(Dy, Dx)
	local hiddenSize = 64
	-- Recogniser
	local input  = - nn.View(-1, Dy)
	local hidden = input
				   - nn.Linear(Dy, hiddenSize)
				   - nn.Tanh()
				   - nn.Linear(hiddenSize, hiddenSize)
				   - nn.Tanh()

	local hy    = hidden
				   - nn.Linear(hiddenSize, Dx)
	local Jy    = hidden
				   - nn.Linear(hiddenSize, Dx)  -- logVar
				   - nn.MulConstant(0.1)
				   - nn.Tanh()
				   - nn.MulConstant(10)
				   - nn.Exp()					-- Var  
				   - nn.MulConstant(-0.5)		-- Jy

	local recogniser = nn.gModule( {input}, {hy, Jy})

	-- LatentGMM
	local phi = - nn.Identity() -- [K, N ]
	local hk  = - nn.Identity() -- [K, Dx]
	local Jk  = - nn.Identity() -- [K, Dx]

	local phiT = phi - nn.Transpose({1,2}) -- [N, K]
	local Ehk  = {phiT, hk} - nn.MM() -- [N, Dx]
	local EJk  = {phiT, Jk} - nn.MM() -- [N, Dx]

	local globalMixing = nn.gModule({phi, hk, Jk}, {Ehk, EJk})
	-- 

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

	local latentGMM = nn.gModule({hy, Jy, Ehk, EJk}, {mean, var})

	-- Sampler
	local mean   = - nn.Identity()
	local var    = - nn.Identity() 
	local rand   = - nn.Identity()
	local std    = var - nn.Power(-1)
	local noise  = {std, rand}
				   - nn.CMulTable()
	local x 	 = {mean, noise}
				   - nn.CAddTable()			   

	local sampler = nn.gModule({mean, var, rand}, {x})

	-- Generator
	local X_sample   = - nn.Identity()
	local h          = X_sample
					   - nn.Linear(Dx, hiddenSize)
					   - nn.Tanh()

	local recon_mean =   h
					   - nn.Linear(hiddenSize, Dy)
	local recon_logVar = h
						- nn.Linear(hiddenSize, Dy) 

	local generator = nn.gModule({X_sample}, {recon_mean, recon_logVar})

	return recogniser, sampler, generator, latentGMM, globalMixing
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
dir:setPrior(100)

-- Initialise parameters
local m1 = torch.rand(K, Dx)
local l1 = torch.Tensor(K, Dx):fill(1)
local a1 = torch.Tensor(K, Dx):fill(1)
local b1 = torch.Tensor(K, Dx):fill(1)
local pi1 = torch.rand(K):fill(10)
NG:setParameters(m1, l1, a1, b1)
dir:setParameters(pi1)

local latentContainer = nn.Container()
						:add(NG)
						:add(dir)
local latentParams, gradLatentParams = latentContainer:getParameters()


local recogniser, sampler, generator, latentGMM, globalMixing = createNetwork(Dy, Dx)
local container = nn.Container()
				  :add(recogniser)
				  :add(generator)
				  

local ReconCrit = nn.GaussianCriterion( batchScale )
local KLCrit    = nn.KLCriterion( 0.0*batchScale )

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
	end
	
	-- Compute Mixture stats
	local Txz = gaussian:getMixtureStats(phi, Tx, batchScale)

	-- Update global parameters
	NG:backward(Txz)
	dir:backward(Txz)

	-- Do sampling
	local rand  = torch.randn(var_x:size())
	local xs    = sampler:forward({mean_x, var_x, rand})
	
	-----------------------

	local recon     = generator:forward(xs)
	local reconLoss = ReconCrit:forward(recon, y)

	local gradRecon = ReconCrit:backward(recon, y)
	local gradXs    = generator:backward(xs, gradRecon)
	local gradMean, gradVar, __ = unpack( sampler:backward({mean_x, var_x, rand}, gradXs ) )
	local gradHy, gradJy, __ = unpack( latentGMM:backward({hy, Jy, Ehk, EJk}, {gradMean, gradVar}) )
	recogniser:backward(y, {gradHy, gradJy})
	

	local var_k = EJk:clone():mul(-2):pow(-1)
	local mean_k = Ehk:clone():cmul(var_k)
	local KLLoss = KLCrit:forward({mean_x, var_x},{mean_k, var_k})
	
	local gradMean, gradVar  = unpack( KLCrit:backward({mean_x, var_x},{mean_k, var_k}) )
	local gradHy, gradJy, __ = unpack( latentGMM:backward({hy, Jy, Ehk, EJk}, {gradMean, gradVar}) )
	recogniser:backward(y, {gradHy, gradJy})
	

	local loss   = reconLoss + KLLoss
	return loss, gradParams
end

function Lfeval(params)
	return __, gradLatentParams
end


for epoch = 1, max_Epoch do

	indices = torch.randperm(N):long():split(batch)

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











