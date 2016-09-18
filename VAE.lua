require 'nn'
require 'GaussianCriterion'
require 'optim'
require 'nngraph'
require 'KLCriterion'

torch.manualSeed(1)
data = torch.load('save/spiral.t7')

local N  = data:size(1)
local Dy = data:size(2)
local Dx = 2
local batch = 100
local batchScale = N/batch
local eta = 0.001
local optimiser = 'adam'
local max_Epoch = 500


-- Network 
function createNetwork(Dy, Dx)
	local hiddenSize = 64
	-- Recogniser
	local input  = - nn.View(-1, Dy)
	local hidden = input
				   - nn.Linear(Dy, hiddenSize)
				   - nn.ReLU(true)
	local mean   = hidden
				   - nn.Linear(hiddenSize, Dx)
	local logVar = hidden
				   - nn.Linear(hiddenSize, Dx)

	local recogniser = nn.gModule( {input}, {mean, logVar})

	-- Sampler
	local mean   = - nn.Identity()
	local std    = - nn.Identity() 
	local rand   = - nn.Identity()
	local noise  = {std, rand}
				   - nn.CMulTable()
	local x 	 = {mean, noise}
				   - nn.CAddTable()			   

	local sampler = nn.gModule({mean, std, rand}, {x})

	-- Generator
	local X_sample   = - nn.Identity()
	local h          = X_sample
					   - nn.Linear(Dx, hiddenSize)
					   - nn.ReLU(true)
	local recon_mean = h
					   - nn.Linear(hiddenSize, Dy)
	local recon_logVar = h
						- nn.Linear(hiddenSize, Dy) 

	local generator = nn.gModule({X_sample}, {recon_mean, recon_logVar})

	return recogniser, sampler, generator
end

local recogniser, sampler, generator = createNetwork(Dy, Dx)
local container = nn.Container()
				  :add(recogniser)
				  :add(generator)

local ReconCrit = nn.GaussianCriterion()
local KLCrit    = nn.KLCriterion()

local params, gradParams = container:getParameters()


function feval(x)

	if x ~= params then
        params:copy(x)
    end

	container:zeroGradParameters()

	local mean, logVar = unpack( recogniser:forward(y) )
	
	local std   = logVar:clone():mul(0.5):exp()
	local rand  = torch.randn(std:size())
	local xs    = sampler:forward({mean, std, rand})

	local recon = generator:forward(xs)
	local reconLoss = ReconCrit:forward(recon, y)
	local gradRecon = ReconCrit:backward(recon, y)
	local gradXs    = generator:backward(xs, gradRecon)
	
	local gradMean, gradStd, __ = unpack( sampler:backward({mean, std, rand}, gradXs) )
	local gradLogVar = gradStd:clone():cmul(std):mul(0.5)

	recogniser:backward(y, {gradMean, gradLogVar})
	
	local var = std:pow(2)
	local KLLoss = KLCrit:forward({mean, var},{})
	local gradMean, gradVar = unpack( KLCrit:backward({mean, var},{}) )
	gradLogVar = torch.cmul(gradVar, var)
	recogniser:backward(y, {gradMean, gradLogVar})
	local loss = reconLoss + KLLoss
	return loss, gradParams
end

for epoch = 1, max_Epoch do

	indices = torch.randperm(N):long():split(batch)

	local recon = torch.Tensor():resizeAs(data):zero()
	local x_sample = torch.Tensor():resize(N, Dx):zero()
	local Loss = 0.0

	for t,v in ipairs(indices) do 
		xlua.progress(t, #indices)
		y = data:index(1,v)
		__, loss = optim[optimiser](feval, params, {learningRate = eta })
		recon[{ { batch*(t-1) + 1, batch*t },{}}]    = generator.output[1]
		x_sample[{ { batch*(t-1) + 1, batch*t },{}}] = sampler.output
		Loss = Loss + loss[1]
	end

	print("Epoch: " .. epoch .. " Loss: " .. Loss/N )
	
	torch.save('save/recon.t7', recon)
	torch.save('save/xs.t7', x_sample)

end 















