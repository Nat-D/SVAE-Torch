require 'nn'
require 'NormalGamma'
require 'Gaussian'
-- Normal variational inference in gaussian with normal-gamma prior

torch.manualSeed(1)
local N = 1000
local D = 2
local K = 1
-- 1.) prepare data and Model
local x = torch.Tensor(N,D):zero()
function generateData()
	local xmean = torch.Tensor{{10}, {20}}
	local cov = torch.Tensor{{10,-0.8},{-0.8,10}}
	local chol = torch.potrf(cov,'L')
	for i=1 , N do
		local rand = torch.randn(D,1)
		x[i]:add(xmean)
		x[i]:add(torch.mm(chol,rand))
	end
end
generateData()

local NG = nn.NormalGamma(K, D)
local gaussian = nn.Gaussian(K, D, N)

-- 2.) set prior 
local m0 = torch.Tensor(D):zero()
local l0 = torch.Tensor(D):fill(1)
local a0 = torch.Tensor(D):fill(1)
local b0 = torch.Tensor(D):fill(1)
local prior = NG:setPrior(m0, l0, a0, b0)

-- 3.) get statistics from data
local Tx = gaussian:observe(x)
-- 5.) Update variational parameters
NG:backward(Tx)
NG:updateParameters(1.0)

local m,l, a, b = unpack( NG:getBasicParameters() )



local x_bar  = x:sum(1)/N
print(m)
print(x_bar)

local cov = torch.Tensor(K, D, D)
for k=1,K do
	cov[k] = torch.diag( torch.cdiv(b,a)[k] )
end
print(cov)
torch.save('save/x.t7', x)
torch.save('save/m.t7', m)
torch.save('save/cov.t7', cov)




