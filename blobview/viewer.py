'''
This is a small little molecule & density viewer that is useful for visually interrogating results of blobBITS calculations. It's not extremely powerful, but also... not meant to be. Written with vispy.

You only need to use the function 'view'
Press h for help.
'''

import time
import numpy as np
import numba as nb

from vispy import app, scene
from vispy.visuals.transforms import STTransform
from vispy.visuals import CompoundVisual, MeshVisual, LineVisual
from vispy.geometry import create_sphere
from vispy.scene.visuals import create_visual_node
from vispy.visuals.shaders import Function
from vispy.io.mesh import write_mesh
from vispy.visuals.filters.base_filter import Filter
from vispy.geometry.isosurface import isosurface
from vispy.geometry.meshdata import MeshData
from vispy.io import write_png

from matplotlib.colors import to_rgba
from matplotlib.cm import tab10
from matplotlib.cm import bwr_r

help_msg = '''
1 - change camera
2 - toggle filters on/off
3 - tight view filters
4 - wide view filters
5 - toggle atom coloring mode
6 - toggle wireframe on/off
7 - toggle polygon surface on/off
8 - save figure (png)
0 - reset camera position
h - display help message
'''

default_options = {
	'bgcolor' : 'w',
	'atom_resolution' : 10,
	'atom_radius' : .25,
	'alpha_iso' : .8,
	'alpha_molecule' : .6,
	'alpha_wire' : .8,
	'fov' : 75.,
	'atom_edge_color' : None,
	'bond_color':(.5,.5,.5,.8),
	'bond_radius':.05,
	'iso_thresh' : .1,
	'iso_color' : (0.3,.3,.3),
	'wire_color' : (0.3,.3,.3),
	'quickbond_cutoff': 1.8,
	'width' : 800,
	'height' : 600,
}

def view(density=None, grid=None, mol=None, probs=None, options=default_options, verbose=True):
	viewer_ = viewer(density, grid, mol, probs, options=options, verbose=verbose)
	app.run()

def colors_by_element(mol,alpha=.9):
	jmol_colors = {
	#JMol
	'C':'#909090',
	'N':'#3050F8',
	'O':'#FF0D0D',
	'S':'#FFFF30',
	# 'H':'#FFFFFF',
	'H':'#444444',
	}
	colors = np.array(['#FFFFFF',]*mol.natoms)
	for element in jmol_colors.keys():
		keep = mol.element == element
		colors[keep] = jmol_colors[element]
	return colors

def colors_by_chain(mol,alpha=.9):

	colors = np.array([(1.,1.,1.,1.),]*mol.natoms)
	uc = mol.unique_chains
	for i in range(uc.size):
		chain = uc[i]
		keep = mol.chain == chain
		colors[keep] = tab10(i)
	colors [:,-1] = alpha
	return colors

def colors_by_array(pnot,alpha=.9):
	p = pnot.copy()
	p[~np.isfinite(p)] = 0.
	colors = bwr_r(p)
	colors [:,-1] = alpha
	return colors

def quick_bonds(mol,cutoff):
	if mol is None:
		return np.array((0,2),dtype=np.int32)

	@nb.njit
	def bondlist(xyz,cutoff):
		nmol,ndim = xyz.shape
		d = np.zeros((nmol*6,2),dtype=nb.int32) ## only so many bonds (octahedral here)
		total_bonds = 0
		ri = np.zeros(3)
		rj = np.zeros(3)
		c2 = cutoff**2.
		for i in range(nmol):
			ri = xyz[i]
			for j in range(i+1,nmol):
				rj = xyz[j]
				if (ri[0]-rj[0])**2. + (ri[1]-rj[1])**2. + (ri[2]-rj[2])**2. < c2:
					d[total_bonds,0] = i
					d[total_bonds,1] = j
					total_bonds += 1
		d = d[:total_bonds]
		return d

	bl = bondlist(mol.xyz,cutoff)

	keep = np.zeros(bl.shape[0],dtype='bool')
	elem1 = mol.element[bl[:,0]]
	elem2 = mol.element[bl[:,1]]
	keep = np.bitwise_not(elem1 == "H",elem2 == "H")
	bl = bl[keep]

	conf1 = mol.conf[bl[:,0]]
	conf2 = mol.conf[bl[:,1]]
	keep = conf1==conf2
	bl = bl[keep]

	return bl

def get_bond_faces(xyz,bonds,radius=.6,nrad=10):
	@nb.njit
	def get_bond_faces_nb(xyz,bonds,r,nrad):
		## inputs xyz,bonds
		natoms = xyz.shape[0]
		nbonds = bonds.shape[0]

		## dims
		p0 = np.zeros(3)
		p1 = np.zeros(3)
		e1 = np.array((1,0,0))
		b0 = np.zeros(3)
		b1 = np.zeros(3)
		b2 = np.zeros(3)
		ps = np.zeros((nrad*2,3))
		pr = np.zeros((nrad,3))
		faces = np.zeros((nrad*2,3),dtype=nb.uint32)
		# x = np.linspace(0,2*np.pi,nrad+1)[:-1,None]
		x = np.arange(nrad+1)/float(nrad+1)*2.*np.pi
		x = x[:-1]

		npoints = faces.shape[0]
		total_verts = np.zeros((nbonds*nrad*2,3))
		total_faces = np.zeros((nbonds*npoints,3),dtype=nb.uint32)

		## setup faces
		for i in range(nrad-1):
			faces[2*i,0] = i
			faces[2*i,1] = i + nrad
			faces[2*i,2] = i + 1 + nrad
			faces[2*i+1,0] = i
			faces[2*i+1,1] = i + 1
			faces[2*i+1,2] = i + 1 + nrad
		faces[-2,0] = nrad - 1
		faces[-2,1] = 2*nrad -1
		faces[-2,2] = nrad
		faces[-1,0] = nrad - 1
		faces[-1,1] = 0
		faces[-1,2] = nrad

		for k in range(nbonds):
			## these two points
			p0 = xyz[bonds[k,0]]
			p1 = xyz[bonds[k,1]]

			## get basis set
			b0 = (p1-p0)
			b0 /= np.sqrt(b0[0]*b0[0]+b0[1]*b0[1]+b0[2]*b0[2])
			b1 = np.cross(e1,b0)
			b1 /= np.sqrt(b1[0]*b1[0]+b1[1]*b1[1]+b1[2]*b1[2])
			b2 = np.cross(b1,b0)
			b2 /= np.sqrt(b2[0]*b2[0]+b2[1]*b2[1]+b2[2]*b2[2])

			## points around circle
			for i in range(x.size):
				pr[i] = r*b1*np.sin(x[i]) + r*b2*np.cos(x[i])
			## circles at each atom
			ps[:nrad] = p0+pr
			ps[nrad:] = p1+pr

			for i in range(npoints):
				total_faces[k*npoints+i] = faces[i]+k*npoints
			for i in range(nrad*2):
				total_verts[k*nrad*2+i] = ps[i]
		return total_verts,total_faces
	return get_bond_faces_nb(xyz,bonds,radius,nrad)

def trace_backbone_protein(mol):
	bonds = []
	if mol is None:
		return bonds

	for chain in mol.unique_chains:
		subchain = mol.get_chain(chain)
		trace = []
		for residue in subchain.unique_residues:
			subres = subchain.get_residue(residue)
			if np.sum((subres.atomname == 'N')+(subres.atomname == 'CA')+(subres.atomname == 'C')) == 3:
				atom_n = subres.get_atomname('N')
				atom_ca = subres.get_atomname('CA')
				atom_c = subres.get_atomname('C')
				trace += [atom_n.xyz[0],atom_ca.xyz[0],atom_c.xyz[0]]
		if len(trace) > 0:
			trace = np.array(trace)
			bonds.append(trace)
	return bonds

# def mask_density_mol(mol,grid,density,fill_value=0,cutoff=4.):
# 	@nb.njit
# 	def _mask_density_mol(xyz,origin,nxyz,dxyz,density,value,cutoff):
# 		out = np.zeros(density.shape)+value
# 		for i in range(nxyz[0]):
# 			x = origin[0] + i*dxyz[0]
# 			for j in range(nxyz[1]):
# 				y = origin[1] + j*dxyz[1]
# 				for k in range(nxyz[2]):
# 					z = origin[2] + k*dxyz[2]
# 					for l in range(xyz.shape[0]):
# 						rl = xyz[l]
# 						r = np.sqrt((rl[0]-x)**2.+(rl[1]-y)**2. + (rl[2]-z)**2.)
# 						if r < cutoff:
# 							out[i,j,k] = density[i,j,k]
# 							break
# 		return out
#
# 	return _mask_density_mol(mol.xyz,grid.origin,grid.nxyz,grid.dxyz,density.copy(),fill_value,cutoff)

class molVisual(CompoundVisual):
	"""
	based off of sphere
	"""
	def __init__(self, xyz, bonds=None, bond_color=(.5,.5,.5,1.), bond_nrad=10, bond_radius=.10/2, atom_radius=1.0, cols=30, rows=30, depth=30, subdivisions=3, method='latitude', vertex_colors=None, face_colors=None, color=(0.5, 0.5, 1, 1), edge_color=None, shading=None, **kwargs):

		self.natom = xyz.shape[0]
		mmesh = create_sphere(rows, cols, depth, radius=atom_radius, subdivisions=subdivisions, method=method)
		vs = mmesh.get_vertices()
		fs = mmesh.get_faces()
		self.nres = fs.shape[0]
		vertices = np.concatenate([vs+xyz[i][None,:] for i in range(xyz.shape[0])],axis=0)
		faces = np.concatenate([fs+vs.shape[0]*i for i in range(xyz.shape[0])],axis=0)

		if not face_colors is None:
			if face_colors.shape[0] != xyz.shape[0]:
				raise Exception('colors malformed length')
			face_colors = np.concatenate([np.repeat(face_colors[i][None,:],fs.shape[0],axis=0) for i in range(self.natom)],axis=0)
		self._mesh = MeshVisual(vertices=vertices,faces=faces,vertex_colors=vertex_colors,face_colors=face_colors, color=color,shading=shading)

		if not edge_color is None:
			es = mmesh.get_edges()
			edges = np.concatenate([es+vs.shape[0]*i for i in range(self.natom)],axis=0)
			self._border = MeshVisual(vertices=vertices, faces=edges,color=edge_color, mode='lines')
		else:
			self._border = MeshVisual()

		if not bonds is None:
			vs,fs = get_bond_faces(xyz,bonds,bond_radius,bond_nrad)
			self._bonds = MeshVisual(vertices=vs,faces=fs,color=bond_color,shading=shading)
		else:
			self._bonds = MeshVisual()

		CompoundVisual.__init__(self, [self._mesh, self._border,self._bonds], **kwargs)
		self._mesh.set_gl_state(blend=True, depth_test=True, polygon_offset_fill=False)

	def update_colors(self,atom_colors):
		face_colors = np.concatenate([np.repeat(atom_colors[i][None,:],self.nres,axis=0) for i in range(self.natom)],axis=0)
		md = self._mesh.mesh_data
		self._mesh.set_data(md._vertices,md._faces,md.get_vertex_colors(),face_colors,md.get_vertex_values())
		self.update()


class densityVisual(CompoundVisual):
	"""
	based off of sphere
	"""
	def __init__(self, vertices = None, faces = None, density=None,  grid=None, level=.5, wireframe=True, iso_color=None,shading=None, **kwargs):

		if not density is None:
			if vertices is None and faces is None:
				self._vertices, self._faces = isosurface(density,level)
			else:
				self._vertices = vertices
				self._faces = faces

			if not grid is None and vertices is None and faces is None:
				self._vertices = self._vertices* grid.dxyz[None,:] + grid.origin[None,:]

			if wireframe:
				self._edges = MeshData(vertices=self._vertices, faces=self._faces).get_edges()
				self._density = MeshVisual()#vertices=self._vertices, faces=self._faces,color=iso_color,shading=shading)
				ls = np.zeros((self._edges.shape[0]*2,3))
				ls[::2] = self._vertices[self._edges[:,0]]
				ls[1::2] = self._vertices[self._edges[:,1]]
				self._densityedge = LineVisual(ls, width=1., color=iso_color, connect='segments',method='gl',antialias=True,)
			else:
				self._density = MeshVisual(vertices=self._vertices, faces=self._faces, color=iso_color,shading=shading)
				self._densityedge = MeshVisual()
				# md = self._density._meshdata
				# write_mesh('surface_mesh.obj',md.get_vertices(),md.get_faces(),md.get_face_normals(),None,overwrite=True)

		else:
			self._density = MeshVisual()
			self._densityedge = MeshVisual()

		CompoundVisual.__init__(self, [self._density,self._densityedge], **kwargs)
		self._density.set_gl_state(blend=True, depth_test=True, polygon_offset_fill=False)

class depthfilter(Filter):
	def __init__(self,
				defval,
				front_cut=15,
				back_cut=15.,
				front_minval=.7,
				back_minval=.01,
				front_rate=1.,
				back_rate=1./3,
				beta = 2.
				):
		Filter.__init__(self)
		self.update_filter(defval,front_cut,back_cut,front_minval,back_minval,front_rate,back_rate,beta)

	def _attach(self, visual):
		visual._get_hook('frag', 'post').add(self.shader())
	def _detach(self, visual):
		## This is a bad hack. Fix it later.
		visual._get_hook('frag', 'post').remove(list(visual._get_hook('frag', 'post').items.keys())[-1])

	def update_filter(self,defval,front_cut,back_cut,front_minval,back_minval,front_rate,back_rate,beta):
		''' two sided power-law decay in distance to camera. '''
		self.shader = Function("""
			uniform vec3 eyePosition;
			void screen_filter() {
				//float z = gl_FragCoord.z / gl_FragCoord.w;
				float z = 1./gl_FragCoord.w;
				float alpha = %f;

				if( z > %f ) {
					alpha *= exp(-pow(%f*(z-%f),%f));
					if (alpha < %f) { alpha=%f;} //{ discard;}
				} else if (z < %f) {
					alpha *= exp(-pow(%f*(%f-z),%f));
					if (alpha < %f) { discard;}
				}

				gl_FragColor[3] = alpha;
			}
		"""%(defval,back_cut,back_rate,back_cut,beta,back_minval,back_minval, front_cut,front_rate,front_cut,beta,front_minval))
		# do it this way so that the overhead is during the construction of the filter not for every gl frag

vismol = create_visual_node(molVisual)
visdensity = create_visual_node(densityVisual)

class viewer(object):
	def __init__(self,data,grid,mol,probs=None,options={},verbose=True):
		self.mol = mol
		self.data = data
		self.grid = grid
		self.probs = probs
		self.options = options
		self.verbose = verbose

		if verbose:
			print(self.options)

		# Prepare canvas
		self.canvas = scene.SceneCanvas(keys='interactive', size=(options['width'],options['height']), show=True,bgcolor=options['bgcolor'],title='blobview')
		self.view = self.canvas.central_widget.add_view()

		self.create_molecule()
		self.create_cameras()
		self.setup_keys()

		self.use_wide_filters()
		self.attach_filters()

		if self.verbose: print(self.help_message())
		self.hook = lambda : None

	def setup_keys(self):
		@self.canvas.events.key_press.connect
		def on_key_press(event):
			if event.text == '1':
				cam_toggle = {self.cam1: self.cam2, self.cam2: self.cam3, self.cam3: self.cam1}
				self.view.camera = cam_toggle.get(self.view.camera, self.cam2)
				if self.verbose: print(self.view.camera.name + ' camera')
			elif event.text == '0':
				self.reset_camera()
			elif event.text == '2':
				if self.flag_filters:
					self.remove_filters()
				else:
					self.attach_filters()
			elif event.text == '3':
				self.remove_filters()
				if self.flag_filter == 'tight':
					self.use_medium_filters()
				elif self.flag_filter == 'medium':
					self.use_wide_filters()
				elif self.flag_filter == 'wide':
					self.use_tight_filters()
				self.attach_filters()

			elif event.text == '4':
				if self.flag_colors == 'probs':
					self.colors_by_element()
				elif self.flag_colors == 'elements':
					self.colors_by_chain()
				elif self.flag_colors == 'chains':
					self.colors_by_prob()
				self.mol_vis.update_colors(self.colors)
			elif event.text == '5':
				self.mol_vis.visible = not self.mol_vis.visible
				self.view.update()
			elif event.text == '6':
				self.wire_vis.visible = not self.wire_vis.visible
				self.view.update()
			elif event.text == '7':
				self.iso_vis.visible = not self.iso_vis.visible
				self.view.update()
			elif event.text == '8':
				fname = 'mol_%s.png'%(str(time.time()))
				cs = self.canvas.native.size()
				self.canvas.native.resize(cs.width()*2,cs.height()*2)
				self.view.update()
				write_png(fname,self.canvas.render())
				self.canvas.native.resize(cs)
				if self.verbose: print('saved',fname)
			elif event.text =='h':
				if self.verbose: print(self.help_message())
			elif event.text == 'q':
				self.hook()

	def help_message(self):
		return help_msg

	def reset_camera(self):
		self.view.camera.depth_value = 1000
		if not self.mol is None:
			self.view.camera.center=self.mol.xyz.mean(0)
		else:
			self.view.camera.center=np.array((0.,0.,0))
		self.view.camera.distance=30

	def create_cameras(self):
		self.cam1 = scene.cameras.FlyCamera(parent=self.view.scene, fov=self.options['fov'], name='Fly',scale_factor=.5)
		self.cam2 = scene.cameras.TurntableCamera(parent=self.view.scene, fov=self.options['fov'],name='Turntable')
		self.cam3 = scene.cameras.ArcballCamera(parent=self.view.scene, fov=self.options['fov'], name='Arcball')
		self.view.camera = self.cam3  # Select turntable at first
		self.reset_camera()

	def create_molecule(self):
		self.colors_by_element()
		self.bonds = quick_bonds(self.mol,self.options['quickbond_cutoff'])

		if not self.mol is None:
			self.mol_vis = vismol(self.mol.xyz,
							bonds = self.bonds,
							parent=self.view.scene,
							bond_radius=self.options['bond_radius'],
							bond_color=self.options['bond_color'],
							atom_radius=self.options['atom_radius'],
							edge_color=self.options['atom_edge_color'],
							cols=self.options['atom_resolution'],
							rows=self.options['atom_resolution'],
							face_colors=self.colors,
							shading=None
							)

		if not self.grid is None:
			self.wire_vis = visdensity(density=self.data,
								grid=self.grid,
								wireframe=True,
								level=self.options['iso_thresh'],
								iso_color=self.options['wire_color'],
								parent=self.view.scene,
								shading=None
								)

			self.iso_vis = visdensity(vertices=self.wire_vis._vertices, ## avoid double isosurface calculation
								faces=self.wire_vis._faces,
								density=self.data,
								grid=self.grid,
								wireframe=False,
								level=self.options['iso_thresh'],
								iso_color=self.options['iso_color'],
								parent=self.view.scene,
								shading='flat'
								)

	def use_tight_filters(self):
		# defval,front_cut,back_cut,front_minval,back_minval,front_rate,back_rate
		fc = 3
		bc = 5.
		fm = .5
		bm = .01
		fr = 5.
		br = 1./3.
		beta = 3.
		self.mol_filter = depthfilter(self.options['alpha_molecule'],fc,bc,fm,bm,fr,br,beta)
		self.wire_filter = depthfilter(self.options['alpha_wire'],fc,bc,fm,bm,fr,br,beta)
		self.iso_filter = depthfilter(self.options['alpha_iso'],fc,bc,fm,bm,fr,br,beta)
		self.flag_filter='tight'
	def use_medium_filters(self):
		# defval,front_cut,back_cut,front_minval,back_minval,front_rate,back_rate
		fc = 5.
		bc = 20.
		fm = .5
		bm = .01
		fr = 5.
		br = 1./10.
		beta = 4.
		self.mol_filter = depthfilter(self.options['alpha_molecule'],fc,bc,fm,bm,fr,br,beta)
		self.wire_filter = depthfilter(self.options['alpha_wire'],fc,bc,fm,bm,fr,br,beta)
		self.iso_filter = depthfilter(self.options['alpha_iso'],fc,bc,fm,bm,fr,br,beta)
		self.flag_filter='medium'
	def use_wide_filters(self):
		# defval,front_cut,back_cut,front_minval,back_minval,front_rate,back_rate
		fc = 10.
		bc = 30.
		fm = .5
		bm = .01
		fr = 5.
		br = 1./30.
		beta = 4.
		self.mol_filter = depthfilter(self.options['alpha_molecule'],fc,bc,fm,bm,fr,br,beta)
		self.wire_filter = depthfilter(self.options['alpha_wire'],fc,bc,fm,bm,fr,br,beta)
		self.iso_filter = depthfilter(self.options['alpha_iso'],fc,bc,fm,bm,fr,br,beta)
		self.flag_filter='wide'

	def attach_filters(self):
		if not self.mol is None:
			self.mol_vis.attach(self.mol_filter)
		if not self.grid is None:
			self.wire_vis.attach(self.wire_filter)
			self.iso_vis.attach(self.iso_filter)
		self.flag_filters = True
		self.view.update()

	def remove_filters(self):
		if not self.mol is None:
			self.mol_vis.detach(self.mol_filter)
		if not self.grid is None:
			self.wire_vis.detach(self.wire_filter)
			self.iso_vis.detach(self.iso_filter)
		self.flag_filters = False
		self.view.update()

	def colors_by_prob(self):
		if self.mol is None:
			return
		self.colors = np.ones((self.mol.xyz.shape[0],4))*self.options['alpha_molecule']
		if not self.probs is None:
			cs = colors_by_array(self.probs)
			for i in range(self.mol.xyz.shape[0]):
				self.colors[i] = to_rgba(cs[i])
		self.flag_colors = 'probs'

	def colors_by_element(self):
		if self.mol is None:
			return
		self.colors = np.ones((self.mol.xyz.shape[0],4))*self.options['alpha_molecule']
		cs = colors_by_element(self.mol)
		for i in range(self.mol.xyz.shape[0]):
			self.colors[i] = to_rgba(cs[i])
		self.flag_colors='elements'

	def colors_by_chain(self):
		if self.mol is None:
			return
		self.colors = np.ones((self.mol.xyz.shape[0],4))*self.options['alpha_molecule']
		cs = colors_by_chain(self.mol)
		for i in range(self.mol.xyz.shape[0]):
			self.colors[i] = to_rgba(cs[i])
		self.flag_colors='chains'
