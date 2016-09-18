local Label, parent = torch.class( 'nn.Label', 'nn.Module' )

function Label:__init(K, N)
	parent.__init(self)
	self.K = K
	self.N = N
	self.phi = torch.Tensor(K, N)

end

function Label:reset()
	
	self.phi:fill(1):div(self.K + 1e-10)

	return self.phi
end


function Label:setParameters(llh, E_dir)
	local K = self.K
	local N = self.N

	-- phi(k,n) = q(z_n = k) = 1/Z exp( energy(z_n = k) )
	-- Z = sum_j energy(z_n = j)

	self.phi:zero()
	self.phi:add(llh):add(E_dir:repeatTensor(N,1):t())
	
	local max = torch.max(self.phi,1)
	self.phi:add(-1, max:expand(K, N))
	self.phi:exp()
	self.phi:cdiv(self.phi:sum(1):expand(K,N) + 1e-10)

	return self.phi
end

function Label:assignLabel()
	local _, idx = self.phi:max(1)
	return idx:view(self.N)
end