import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def mirror(array):
	temp = []
	i = len(array)-1
	while i > 0 :
		try:
			t = temp[-1]
			temp.append(t+(array[i]-array[i-1]))
		except:
			temp.append(array[i]+(array[i]-array[i-1]))
		i-=1
	array = array.tolist()
	array.extend(temp)    
	return np.array(array)

def input_plot(x_pos, y_pos, mesh):

	plt.set_cmap('Reds')
	plt.pcolormesh(x_pos, y_pos, np.flipud(mesh))
	plt.colorbar()
	plt.savefig('Input.png')

def output_plot(x_pos, y_pos, flux):

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x_pos_full = mirror(x_pos)
	y_pos_full = mirror(y_pos)

	X, Y = np.meshgrid(x_pos_full,  y_pos_full)
	left_bottom = np.reshape(flux,(len(y_pos),len(x_pos)))
	right_bottom = np.fliplr(left_bottom[:,:-1])
	left_top = np.flipud(left_bottom[:-1,:])
	right_top = np.fliplr(left_top[:,:-1])
	top = np.hstack((left_bottom,right_bottom))
	bottom = np.hstack((left_top,right_top))
	full_grid = np.vstack((np.hstack((left_bottom,right_bottom)),np.hstack((left_top,right_top))))

	Z = full_grid
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('flux')
	plt.savefig('3doutput.png',dpi=800) 