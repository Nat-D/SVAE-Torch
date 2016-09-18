local Gaussian, parent  = torch.class('nn.Gaussian', 'nn.Module')

function Gaussian:__init(K, D, N)
	parent.__init(self)
	self.K = K
	self.D = D 
	self.N = N
	self.stats = torch.Tensor(4, K, D)
	self.llh = torch.Tensor(K, N)
end

function Gaussian:observe(data)

	local D = self.D

	-- data [ NxD ]
	self.stats[1][1]:copy( data:sum(1) )
	self.stats[2][1]:copy( torch.cmul(data,data):sum(1) )
	self.stats[3][1]:fill( data:size(1) )
	self.stats[4][1]:fill( data:size(1) )
	
	return self.stats
end

function Gaussian:getLogLikelihood(E_NG, Tx)
	local K = self.K
	local N = self.N 
	local D = self.D

	self.llh:zero() -- [K, N]
	-- log P(x|k, mu, Sig) = < t(g,m), (x, x2, 1 ,1 ) >  
	-- t(g,m): [4, K, D]
	-- Tx[1]: [N, D]
	-- Tx[2]: [N, D]
	self.llh:addmm( E_NG[1], Tx[1]:t() )
	        :addmm( E_NG[2], Tx[2]:t() )
	        :add(E_NG[3]:sum(2):expand(K,N))
	        :add(E_NG[4]:sum(2):expand(K,N))
	        :add(-0.5*torch.log(2*math.pi))
	
	return self.llh
end

function Gaussian:getMixtureStats(phi, Tx, scale)
	local K = self.K
	local N = self.N 
	local D = self.D
	-- txz = sum_n [phi x, phi x2 , phi , phi]  [KxD]
	-- phi [KxN]
	-- Tx [NxD]
	self.stats:zero()
	self.stats[1]:mm(phi, Tx[1])
	self.stats[2]:mm(phi, Tx[2])
	self.stats[3]:copy( phi:sum(2):expand(K,D) )
	self.stats[4]:copy( self.stats[3] )

	self.stats[1]:mul(scale)
	self.stats[2]:mul(scale)
	self.stats[3]:mul(scale)
	self.stats[4]:mul(scale)

	return self.stats
end









