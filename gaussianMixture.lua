require 'nn'
require 'Gaussian'
require 'NormalGamma'
require 'Dirichlet'
require 'Label'

torch.manualSeed(1)

local N = 1000
local D = 2
local K = 20

-- 1.) prepare data 

local function generateMixtureOfGaussian(N, xmean, cov, K)
	local D = xmean:size(2)
	local _x = torch.Tensor(N, D):zero()
	local k = torch.ceil( torch.rand(N)*K )
	for i= 1 , N do
		local chol = torch.potrf(cov[k[i]] , 'L')
		local rand = torch.randn(D,1)
		_x[i]:add(xmean[k[i]])
		_x[i]:add(torch.mm(chol,rand))
	end

	return _x
end

local xmean = 100 * torch.rand(K, D, 1)--torch.Tensor{{10}, {20}}
local cov = 1.0 * torch.Tensor{{1,-0.5},{-0.5,1}}:repeatTensor(K, 1, 1)

local x = generateMixtureOfGaussian(N, xmean, cov, K)


local x = torch.load('save/spiral.t7')
local N = x:size(1)
local D = 2
local K = 5

-- prepare model
local dir = nn.Dirichlet(K)
local NG = nn.NormalGamma(K,D)
local gaussian = nn.Gaussian(K, D, N)
local label = nn.Label(K, N)

-- set prior
local m0 = torch.Tensor(D):zero()
local l0 = torch.Tensor(D):fill(1)
local a0 = torch.Tensor(D):fill(1)
local b0 = torch.Tensor(D):fill(1)
NG:setPrior(m0, l0, a0, b0)
dir:setPrior(1000)

-- Initialise parameters
local m1 = torch.rand(K, D)
local l1 = torch.Tensor(K, D):fill(1)
local a1 = torch.Tensor(K, D):fill(1)
local b1 = torch.Tensor(K, D):fill(1)
local pi1 = torch.rand(K)
NG:setParameters(m1, l1, a1, b1)
dir:setParameters(pi1)

-- prepare x stats
local Tx = { x, torch.cmul(x,x)  }

for epoch =1, 100 do
	NG:zeroGradParameters()
	dir:zeroGradParameters()
	--1.) Get global expected stats
	local E_NG = NG:updateExpectedStats()
	local E_dir = dir:updateExpectedStats()

	--2.) Get gaussian expected llh
	local llh = gaussian:getLogLikelihood(E_NG, Tx)

	--3.) Update Label parameters
	local phi = label:setParameters(llh, E_dir)

	--4.) Compute Mixture stats
	local Txz = gaussian:getMixtureStats(phi, Tx, 1.0)

	-- 5.) Update global parameters
	NG:backward(Txz)
	NG:updateParameters(1.0)
	dir:backward(Txz)
	dir:updateParameters(1.0)

	-- plot
	local m, l, a, b = unpack(NG:getBasicParameters())
	local cov = torch.Tensor(K, D, D)
	for k=1,K do
		cov[k] = torch.diag( torch.cdiv(b,a)[k] )
	end
	print(cov)
	torch.save('save/x.t7', x)
	torch.save('save/m.t7', m)
	torch.save('save/cov.t7', cov)

end




































