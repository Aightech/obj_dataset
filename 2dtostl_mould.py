"""
Author: Alexis Devillard
Date: 2023-06-27
Description: This script is used to generate the surfaces with the different frequencies of noise and pattern. The surfaces are saved as .npy, .png and .stl files.
"""

import numpy as np
import scipy.fftpack as fp
import matplotlib.pyplot as plt
from stl import mesh
# import the liv cv2
import cv2
import os

def create_2d_butterworth(image_size, low_freq, high_freq, order):
    # meshgrid to create a 2d array of x and y coordinates
    X, Y = np.meshgrid(range(image_size[1]), range(image_size[0]))
    X = X - (image_size[1] // 2)  # center the x coordinates
    Y = Y - (image_size[0] // 2)  # center the y coordinates
    radius = np.sqrt(X**2 + Y**2)  # calculate the radius of each point
    # shift the radius so that the center is at 0,0
    radius = np.fft.fftshift(radius)
    lowpass = 1 / (1 + (radius / low_freq)**(2 * order))
    highpass = 1 / (1 + (high_freq / radius)**(2 * order))
    return lowpass * highpass

def np2stl_thin_surface(mesh_x, mesh_y, mesh_z, res=1, thickness=0.002):
    scale_factor = np.array([1/res, 1/res, 1])*1000
    mesh_x = mesh_x * scale_factor[0]
    mesh_y = mesh_y * scale_factor[1]
    mesh_z = mesh_z * scale_factor[2]
    image_size = mesh_x.shape
    
    # Convert the mesh to an STL file
    num_faces = 4 * (image_size[0] - 1) * (image_size[1] - 1) +  4*(image_size[0] + image_size[1] - 2)
    output_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))

    k=0
    thickness = thickness * scale_factor[2]
    
    # create two triangle between each square of points and 2 more triange offset by the thickness on the z axis
    for i in range(image_size[0] - 1):
        
        for j in range(image_size[1] - 1):
            p1 = [mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]]
            p2 = [mesh_x[i + 1, j], mesh_y[i + 1, j], mesh_z[i + 1, j]]
            p3 = [mesh_x[i + 1, j + 1], mesh_y[i + 1, j + 1], mesh_z[i + 1, j + 1]]
            p4 = [mesh_x[i, j + 1], mesh_y[i, j + 1], mesh_z[i, j + 1]]
            # Create the two triangular faces
            output_mesh.vectors[k] = [p1, p2, p3]
            output_mesh.vectors[k+1] = [p1, p3, p4]
            # Create the two triangular faces offset by the thickness
            output_mesh.vectors[k+2] = [p1+np.array([0,0,thickness]), p2+np.array([0,0,thickness]), p3+np.array([0,0,thickness])]
            output_mesh.vectors[k+3] = [p1+np.array([0,0,thickness]), p3+np.array([0,0,thickness]), p4+np.array([0,0,thickness])]
            k+=4
    # for all border points create a base a triangle between the point and the bottom vertex of the base
    for i in range(image_size[0] - 1):
        p1 = [mesh_x[i, 0], mesh_y[i, 0], mesh_z[i, 0]]
        p2 = [mesh_x[i + 1, 0], mesh_y[i + 1, 0], mesh_z[i + 1, 0]]
        p3 = [mesh_x[i, 0], mesh_y[i, 0], mesh_z[i, 0]+thickness]
        p4 = [mesh_x[i + 1, 0], mesh_y[i + 1, 0], mesh_z[i + 1, 0]+thickness]
        # Create the two triangular faces
        output_mesh.vectors[k] = [p1, p2, p3]
        output_mesh.vectors[k+1] = [p2, p3, p4]

        #oppisite side
        p1 = [mesh_x[i, image_size[1]-1], mesh_y[i, image_size[1]-1], mesh_z[i, image_size[1]-1]]
        p2 = [mesh_x[i + 1, image_size[1]-1], mesh_y[i + 1, image_size[1]-1], mesh_z[i + 1, image_size[1]-1]]
        p3 = [mesh_x[i, image_size[1]-1], mesh_y[i, image_size[1]-1], mesh_z[i, image_size[1]-1]+thickness]
        p4 = [mesh_x[i + 1, image_size[1]-1], mesh_y[i + 1, image_size[1]-1], mesh_z[i + 1, image_size[1]-1]+thickness]
        # Create the two triangular faces
        output_mesh.vectors[k+2] = [p1, p2, p3]
        output_mesh.vectors[k+3] = [p2, p3, p4]
        k+=4

    for j in range(image_size[1] - 1):
        p1 = [mesh_x[0, j], mesh_y[0, j], mesh_z[0, j]]
        p2 = [mesh_x[0, j+1], mesh_y[0, j+1], mesh_z[0, j+1]]
        p3 = [mesh_x[0, j], mesh_y[0, j], mesh_z[0, j]+thickness]
        p4 = [mesh_x[0, j+1], mesh_y[0, j+1], mesh_z[0, j+1]+thickness]
        # Create the two triangular faces
        output_mesh.vectors[k] = [p1, p2, p3]
        output_mesh.vectors[k+1] = [p2, p3, p4]

        #oppisite side
        p1 = [mesh_x[image_size[0]-1, j], mesh_y[image_size[0]-1, j], mesh_z[image_size[0]-1, j]]
        p2 = [mesh_x[image_size[0]-1, j+1], mesh_y[image_size[0]-1, j+1], mesh_z[image_size[0]-1, j+1]]
        p3 = [mesh_x[image_size[0]-1, j], mesh_y[image_size[0]-1, j], mesh_z[image_size[0]-1, j]+thickness]
        p4 = [mesh_x[image_size[0]-1, j+1], mesh_y[image_size[0]-1, j+1], mesh_z[image_size[0]-1, j+1]+thickness]
        # Create the two triangular faces
        output_mesh.vectors[k+2] = [p1, p2, p3]
        output_mesh.vectors[k+3] = [p2, p3, p4]
        k+=4

    #return the mesh
    return output_mesh

def np2stl(mesh_x, mesh_y, mesh_z, res=1, thickness=0.002, label="test"):
    scale_factor = np.array([1/res, 1/res, 1])*1000
    mesh_x = mesh_x * scale_factor[0]
    mesh_y = mesh_y * scale_factor[1]
    mesh_z = mesh_z * scale_factor[2]
    image_size = mesh_x.shape
    # create a image of the size of the surface
    label_size = (25, 180)
    img = np.zeros(label_size)
    cv2.putText(
        img, label, (label_size[1], 2), cv2.FONT_HERSHEY_SIMPLEX, -0.9, (255, 255, 255), 2)
    img /= 255
    d=1
    # Convert the mesh to an STL file
    num_faces = (image_size[0] - 1) * (image_size[1] - 1) * \
        2 + 2 * (image_size[0] + image_size[1] - 2)*2+2*4 + label_size[0]*label_size[1]*2
    output_mesh = mesh.Mesh(np.zeros(num_faces, dtype=mesh.Mesh.dtype))
    # create two triangle between each square of points
    for i in range(image_size[0] - 1):
        for j in range(image_size[1] - 1):
            k = (i * (image_size[1] - 1) + j) * 2
            p1 = [mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]]
            p2 = [mesh_x[i + 1, j], mesh_y[i + 1, j], mesh_z[i + 1, j]]
            p3 = [mesh_x[i + 1, j + 1], mesh_y[i + 1, j + 1], mesh_z[i + 1, j + 1]]
            p4 = [mesh_x[i, j + 1], mesh_y[i, j + 1], mesh_z[i, j + 1]]
            # Create the two triangular faces
            output_mesh.vectors[k] = [p1, p2, p3]
            output_mesh.vectors[k + 1] = [p1, p3, p4]
    # for all border points create a base a triangle between the point and the bottom vertex of the base
    # base thickness
    base_thickness = thickness * 1000
    # length of the base
    k = 2*(image_size[0]-1)*(image_size[1]-1)
    for j in [0, image_size[1]-1]:
        for i in range(image_size[0]-1):
            p1 = [mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]]
            p2 = [mesh_x[i, j], mesh_y[i, j], -base_thickness]
            p3 = [mesh_x[i+1, j], mesh_y[i+1, j], -base_thickness]
            p4 = [mesh_x[i+1, j], mesh_y[i+1, j], mesh_z[i+1, j]]
            output_mesh.vectors[k] = [p1, p2, p3]
            output_mesh.vectors[k+1] = [p1, p3, p4]
            k += 2
    # width of the base
    for i in [0, image_size[0]-1]:
        for j in range(image_size[1]-1):
            p1 = [mesh_x[i, j], mesh_y[i, j], mesh_z[i, j]]
            p2 = [mesh_x[i, j], mesh_y[i, j], -base_thickness]
            p3 = [mesh_x[i, j+1], mesh_y[i, j+1], -base_thickness]
            p4 = [mesh_x[i, j+1], mesh_y[i, j+1], mesh_z[i, j+1]]

            output_mesh.vectors[k] = [p1, p2, p3]
            output_mesh.vectors[k+1] = [p1, p3, p4]
            k += 2
    # bottom of the base
    index1 = int(mesh_x.shape[1]/10)
    index2 = int(mesh_x.shape[1]/10+label_size[1]-1)
    index3 = int(mesh_x.shape[1]/10+label_size[0]-1)

    # label square
    l_square = [[mesh_x[0, index1], mesh_y[index1, 0], -base_thickness], [mesh_x[0, index1], mesh_y[index3, 0], -base_thickness],
                [mesh_x[0, index2], mesh_y[index3, 0], -base_thickness], [mesh_x[0, index2], mesh_y[index1, 0], -base_thickness]]
    base_square = [[mesh_x[0, 0], mesh_y[0, 0], -base_thickness], [mesh_x[0, 0], mesh_y[-1, -1], -base_thickness],
                   [mesh_x[-1, -1], mesh_y[-1, -1], -base_thickness], [mesh_x[-1, -1], mesh_y[0, 0], -base_thickness]]
    
    for i in range(4):
        p1 = base_square[i]
        p2 = base_square[(i+1) % 4]
        p3 = l_square[i]
        p4 = l_square[(i+1) % 4]
        output_mesh.vectors[k] = [p1, p2, p3]
        output_mesh.vectors[k+1] = [p2, p3, p4]
        k += 2

    #add label
    for i in range(label_size[0]-1):
        for j in range(label_size[1]-1):
            p1 = [mesh_x[0, index1+j], mesh_y[index1+i, 0], -base_thickness+img[i, j]*d]
            p2 = [mesh_x[0, index1+j], mesh_y[index1+i+1, 0], -base_thickness+img[i+1, j]*d]
            p3 = [mesh_x[0, index1+j+1], mesh_y[index1+i+1, 0], -base_thickness+img[i+1, j+1]*d]
            p4 = [mesh_x[0, index1+j+1], mesh_y[index1+i, 0], -base_thickness+img[i, j+1]*d]
            output_mesh.vectors[k] = [p1, p2, p3]
            output_mesh.vectors[k+1] = [p1, p3, p4]
            k += 2
    

    #return the mesh
    return output_mesh


def create_cubeShell_stl(size, border_thickness):
    '''Create a cube shell with an opening on the top'''
    # Create the mesh
    output_mesh = mesh.Mesh(np.zeros(28, dtype=mesh.Mesh.dtype))
    # outter cube (4 sides + bottom)*2 triangles per side + 8 triangles for the top
    # inner cube (4 sides + bottom)*2 triangles per side

    size = 1000*np.array(size)
    border_thickness = 1000*border_thickness
    print(size)

    # Create 8 points for the outter cube
    points_outter = np.zeros((8, 3)) 
    points_outter[0] = [0, 0, 0] # bottom left
    points_outter[1] = [0, size[1], 0] # top left
    points_outter[2] = [size[0], size[1], 0] # top right
    points_outter[3] = [size[0], 0, 0] # bottom right
    points_outter[4] = [0, 0, size[2]] # bottom left
    points_outter[5] = [0, size[1], size[2]] # top left
    points_outter[6] = [size[0], size[1], size[2]] # top right
    points_outter[7] = [size[0], 0, size[2]] # bottom right

    # Create 8 points for the inner cube
    points_inner = np.zeros((8, 3))
    points_inner[0] = [border_thickness, border_thickness, border_thickness] # bottom left
    points_inner[1] = [border_thickness, size[1]-border_thickness, border_thickness] # top left
    points_inner[2] = [size[0]-border_thickness, size[1]-border_thickness, border_thickness] # top right
    points_inner[3] = [size[0]-border_thickness, border_thickness, border_thickness] # bottom right
    points_inner[4] = [border_thickness, border_thickness, size[2]] # bottom left
    points_inner[5] = [border_thickness, size[1]-border_thickness, size[2]] # top left
    points_inner[6] = [size[0]-border_thickness, size[1]-border_thickness, size[2]] # top right
    points_inner[7] = [size[0]-border_thickness, border_thickness, size[2]] # bottom right

    #create outter cube (4 sides + bottom)*2 triangles
    #bottom
    output_mesh.vectors[0] = [points_outter[0], points_outter[1], points_outter[2]]
    output_mesh.vectors[1] = [points_outter[0], points_outter[2], points_outter[3]]
    #side 1
    output_mesh.vectors[2] = [points_outter[0], points_outter[1], points_outter[5]]
    output_mesh.vectors[3] = [points_outter[0], points_outter[5], points_outter[4]]
    #side 2
    output_mesh.vectors[4] = [points_outter[1], points_outter[2], points_outter[6]]
    output_mesh.vectors[5] = [points_outter[1], points_outter[6], points_outter[5]]
    #side 3
    output_mesh.vectors[6] = [points_outter[2], points_outter[3], points_outter[7]]
    output_mesh.vectors[7] = [points_outter[2], points_outter[7], points_outter[6]]
    #side 4
    output_mesh.vectors[8] = [points_outter[3], points_outter[0], points_outter[4]]
    output_mesh.vectors[9] = [points_outter[3], points_outter[4], points_outter[7]]


    #create inner cube (4 sides + bottom)*2 triangles
    #bottom
    output_mesh.vectors[10] = [points_inner[0], points_inner[1], points_inner[2]]
    output_mesh.vectors[11] = [points_inner[0], points_inner[2], points_inner[3]]
    #side 1
    output_mesh.vectors[12] = [points_inner[0], points_inner[1], points_inner[5]]
    output_mesh.vectors[13] = [points_inner[0], points_inner[5], points_inner[4]]
    #side 2
    output_mesh.vectors[14] = [points_inner[1], points_inner[2], points_inner[6]]
    output_mesh.vectors[15] = [points_inner[1], points_inner[6], points_inner[5]]
    #side 3
    output_mesh.vectors[16] = [points_inner[2], points_inner[3], points_inner[7]]
    output_mesh.vectors[17] = [points_inner[2], points_inner[7], points_inner[6]]
    #side 4
    output_mesh.vectors[18] = [points_inner[3], points_inner[0], points_inner[4]]
    output_mesh.vectors[19] = [points_inner[3], points_inner[4], points_inner[7]]

    #create top
    output_mesh.vectors[20] = [points_outter[4], points_outter[5], points_inner[4]]
    output_mesh.vectors[21] = [points_outter[5], points_inner[5], points_inner[4]]
    output_mesh.vectors[22] = [points_outter[5], points_outter[6], points_inner[5]]
    output_mesh.vectors[23] = [points_outter[6], points_inner[6], points_inner[5]]
    output_mesh.vectors[24] = [points_outter[6], points_outter[7], points_inner[6]]
    output_mesh.vectors[25] = [points_outter[7], points_inner[7], points_inner[6]]
    output_mesh.vectors[26] = [points_outter[7], points_outter[4], points_inner[7]]
    output_mesh.vectors[27] = [points_outter[4], points_inner[4], points_inner[7]]

    return output_mesh




                       

    


def generate_filtered_surface(size_surface, res, low, high, order):
    # create a square with the largest side to ensure uniforme filtering
    max_dim = max(size_surface)
    image_size = (int(max_dim*res), int(max_dim*res))
    noise = np.random.randn(*image_size)    # Generate random noise
    # Define the frequency range (in 1/meters)
    low_freq = image_size[0] * (low * res/10000000)/2
    high_freq = image_size[0] * (high * res/10000000)/2
    # Create a 2D Butterworth bandpass filter
    filter_2D = create_2d_butterworth(image_size, low_freq, high_freq, order)
    # Apply the filter to the noise
    filtered_noise = np.real(fp.ifft2(fp.fft2(noise) * filter_2D))
    # Generate the surface
    surface = filtered_noise
    # Normalize the surface
    surface -= surface.min()
    surface /= surface.max()
    # Generate a 3D mesh from the filtered noise image
    mesh_x, mesh_y = np.meshgrid(
        range(int(size_surface[0]*res)), range(int(size_surface[1]*res)))
    surface = filtered_noise[:int(
        size_surface[1]*res), :int(size_surface[0]*res)]

    return mesh_x, mesh_y, surface

def generate_image(freq, amp, size_surface, res):
    # Define the frequency range (in 1/meters)
    low = freq - 2.5
    high = freq + 2.5
    order = 10
    # Generate a 3D mesh from the filtered noise image
    mesh_x, mesh_y, mesh_z = generate_filtered_surface(
        size_surface, res, low, high, order)

    return mesh_z*amp

def save_image(surface, name):
    # save the mesh_z as a .npy file and as a .png file
    np.save('./data/npy/'+name+'.npy', surface)
    plt.imsave('data/img/'+name+'.png', surface, cmap='gray')



if __name__ == '__main__':
    # resolution 1pixel=0.2mm
    res = 1/(0.2*10**-3)
    size_surface = (0.05, 0.05)  # size of the surface in meters

    list_freq = [10, 20, 30, 40, 50]
    list_amp = [0.02, 0.01, 0.005]
    list_thickness = [0.0012]

    dict_surface = {}

    #for each combination of frequency and amplitude, check if there is an existing npy file
    #if not, generate the surface and save it
    #if yes, add it to the list of surfaces
    for i, freq in enumerate(list_freq):
        for j, amp in enumerate(list_amp):
            name= "F"+str(freq)+"A"+str(int(amp*1000))
            path = 'data/npy/surface_' + name +'.npy'
            if not os.path.exists(path):
                print("generate surface : ", name)
                surface = generate_image(freq, amp, size_surface, res)
                save_image(surface, name)
            else:
                print("load surface : ", name)
                surface = np.load(path)
            dict_surface[name] = surface
            #print the max and min of the surface
    
    #for each surfaces in dict_surface generate an stl
    #iterate over the dict_surface
    for key, surface in dict_surface.items():
        for i, thickness in enumerate(list_thickness):
            name = key+"T"+str(int(thickness*1000))
            path = 'data/stl/thin_surface_' + name +'.stl'
            #if the stl does not exist, generate it and save it
            if not os.path.exists(path):
                print("generate stl : ", name)
                mesh_x, mesh_y = np.meshgrid(
                    range(int(size_surface[0]*res)), range(int(size_surface[1]*res)))
                stl_mesh = np2stl_thin_surface(mesh_x, mesh_y, surface, res, thickness)
                stl_mesh.save(path)
