# Import libraries
from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import cv2

dataset_path = 'D:/Dataset/trainingset/'
dataset_dir = os.listdir( dataset_path )
width = 195
height = 231

print( 'Train Images:' )
train_image_names = [ 'subject01.normal.png', 'subject02.normal.png', 'subject03.normal.png','subject04.normal.png', 'subject05.normal.png','subject06.normal.png','subject07.normal.png','subject08.normal.png','subject09.normal.png', 'subject10.normal.png', 'subject11.normal.png','subject12.normal.png','subject13.normal.png', 'subject14.normal.png', 'subject15.normal.png' ]
training_tensor = np.ndarray( shape=( len( train_image_names ), height * width ), dtype=np.float64 )

for i in range( len( train_image_names ) ):
    img = plt.imread( dataset_path + train_image_names[ i ] )
    training_tensor[ i,: ] = np.array( img, dtype='float64' ).flatten( )
    plt.subplot( 3,5,1 + i )
    plt.title( train_image_names[ i ].split( '.' )[ 0 ] )
    plt.imshow( img,cmap='Greys_r' )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
plt.show( )

# Read Testing Images
testing_dataset_path = 'D:/Dataset/testset/'
testing_dataset_dir = os.listdir( testing_dataset_path )
testing_filelist = [ ]
for filename in testing_dataset_dir:  # Read Filename
    testing_filelist.append( filename )
print( 'Testing Images:' )
testing_tensor = np.ndarray( shape=( 10, height * width ), dtype=np.float64 )# two-dimensional array, i represents test picture, j represents pixel
for i in range( 10 ):
    imgpath = testing_dataset_path + testing_filelist[ i ]
    img = plt.imread( imgpath,cv2.IMREAD_GRAYSCALE )
    testing_tensor[ i ] = np.array( img, dtype='float64' ).flatten( )
    plt.subplot( 2,5,1 + i )
    plt.title( testing_filelist[ i ].split( '.' )[ 0 ] )
    plt.imshow( img, cmap='gray' )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
plt.show( )  #show image

#Mean Face
mean_face = np.zeros( ( 1,height * width ) )
for i in training_tensor:
    mean_face = np.add( mean_face,i )
mean_face = np.divide( mean_face,float( len( train_image_names ) ) ).flatten( )
plt.imshow( mean_face.reshape( height, width ), cmap='gray' )
plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
plt.show( )

#Normalised faces
normalised_training_tensor = np.ndarray( shape=( len( train_image_names ), height * width ) )
for i in range( len( train_image_names ) ):
    normalised_training_tensor[ i ] = np.subtract( training_tensor[ i ],mean_face )

#Display normalised faces
for i in range( len( train_image_names ) ):
    img = normalised_training_tensor[ i ].reshape( height,width )
    plt.subplot( 3,5,1 + i )
    plt.imshow( img, cmap='gray' )
    plt.title( train_image_names[ i ].split( '.' )[ 0 ] )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
plt.show( )

#Covariance matrix
cov_matrix = np.cov( normalised_training_tensor )
cov_matrix = np.divide( cov_matrix,15.0 )
np.set_printoptions(precision=2)
print( 'Covariance matrix of X: \n%s' % cov_matrix )
 
eigenvalues, eigenvectors, = np.linalg.eig( cov_matrix )
print( 'Eigenvectors of Cov(X): \n%s' % eigenvectors )
print( '\nEigenvalues of Cov(X): \n%s' % eigenvalues )

eig_pairs = [( eigenvalues[ index ], eigenvectors[ :,index ] ) for index in range( len( eigenvalues ) )]

# Sort the eigen pairs in descending order:
eig_pairs.sort( reverse=True )
eigvalues_sort = [eig_pairs[ index ][ 0 ] for index in range( len( eigenvalues ) )]
eigvectors_sort = [eig_pairs[ index ][ 1 ] for index in range( len( eigenvalues ) )]

var_comp_sum = np.cumsum( eigvalues_sort ) / sum( eigvalues_sort )

# Show cumulative proportion of varaince with respect to components
print( "Cumulative proportion of variance explained vector: \n%s" % var_comp_sum )

# x-axis for number of principal components kept
num_comp = range( 1,len( eigvalues_sort ) + 1 )
plt.title( 'Cum. Prop. Variance Explain and Components Kept' )
plt.xlabel( 'Principal Components' )
plt.ylabel( 'Cum. Prop. Variance Expalined' )
plt.scatter( num_comp, var_comp_sum )
plt.show( )

#Choose the necessary no.of principle components:
reduced_data = np.array( eigvectors_sort[ :14 ] ).transpose( )

#Now we try to find the projected data.  This will form the eigen space.
proj_data = np.dot( training_tensor.transpose( ),reduced_data )
proj_data = proj_data.transpose( )
#Plot eigen faces
for i in range( proj_data.shape[ 0 ] ):
    img = proj_data[ i ].reshape( height,width )
    plt.subplot( 3,5,1 + i )
    plt.imshow( img, cmap='jet' )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
plt.show( )

#Finding weights for each traning image
w = np.array( [np.dot( proj_data,i ) for i in normalised_training_tensor] )
print( w )

#Now we recognise unknown face!
unknown_face = plt.imread(dataset_path+'subject12.normal.png')
unknown_face_vector = np.array(unknown_face, dtype='float64').flatten()

plt.imshow(unknown_face, cmap='gray')
plt.title('Unknown face')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()

normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
plt.imshow(normalised_uface_vector.reshape(height, width), cmap='gray')
plt.title('Normalised unknown face')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()

w_unknown = np.dot(proj_data, unknown_face_vector)
print(w_unknown)
diff  = w - w_unknown
norms = np.linalg.norm(diff, axis=1)
print(norms)
norm=min(norms)
print(norm)
count = 0
def recogniser ( img, train_image_names,proj_data,w ):
    global count
    unknown_face = plt.imread( testing_dataset_path + img )
    unknown_face_vector = np.array( unknown_face, dtype='float64' ).flatten( )
    normalised_uface_vector = np.subtract( unknown_face_vector,mean_face )
    
    plt.subplot( 5,4,1 + count )
    plt.imshow( unknown_face, cmap='gray' )
    plt.title( 'Input:' + '.'.join( img.split( '.' )[ :2 ] ) )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
    count+=1
    
    plt.subplot( 5,4,1 + count )
    plt.imshow( normalised_uface_vector.reshape( height, width ), cmap='gray' )
    plt.title( 'Normalised Face' )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
    count+=1
for i in range( len( testing_filelist ) ):
    recogniser( testing_filelist[ i ], train_image_names,proj_data,w )
plt.show( )

count = 0
num_images = 0
correct_pred = 0
def recogniser ( img, train_image_names,proj_data,w ):
    global count,highest_min,num_images,correct_pred
    unknown_face = plt.imread( testing_dataset_path + img )
    num_images          += 1
    unknown_face_vector = np.array( unknown_face, dtype='float64' ).flatten( )
    normalised_uface_vector = np.subtract( unknown_face_vector,mean_face )
    
    plt.subplot( 5,4,1 + count )
    plt.imshow( unknown_face, cmap='gray' )
    plt.title( '.'.join( img.split( '.' )[ :2 ] ) )
    plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
    count+=1
    
    w_unknown = np.dot( proj_data, normalised_uface_vector )
    diff = w - w_unknown
    norms = np.linalg.norm( diff, axis=1 )
    index = np.argmin( norms )
    
    t1 = 100111536
    #t1 = 200535910.268 # working with 6 faces
    #t0 = 86528212
    t0 = 88831687
    #t0 = 143559033 # working with 6 faces
    if norms[ index ] < t1:
        plt.subplot( 5,4,1 + count )
        if norms[ index ] < t0: # It's a face
            if img.split( '.' )[ 0 ] == train_image_names[ index ].split( '.' )[ 0 ]:
                plt.title( 'Correctly Matched' + '.'.join( train_image_names[ index ].split( '.' )[ :2 ] ), color='g') 
                plt.imshow( imread( dataset_path + train_image_names[ index ] ), cmap='gray' )
                
                correct_pred += 1
            else:
                plt.title( 'Incorrectly Matched' + '.'.join( train_image_names[ index ].split( '.' )[ :2 ] ), color='r')
                plt.imshow( imread( dataset_path + train_image_names[ index ] ), cmap='gray' )
        else:
            if img.split( '.' )[ 0 ] not in [i.split( '.' )[ 0 ] for i in train_image_names] and img.split( '.' )[ 0 ] != 'apple':
                plt.title( 'Unknown face!', color='g' )
                correct_pred += 1
            else:
                plt.title( 'Unknown face!', color='r' )
        plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
    else:     
        plt.subplot( 5,4,1 + count )
        if len( img.split( '.' ) ) == 3:
            plt.title( 'Not a face!', color='r' )
        else:
            plt.title( 'Not a face!', color='g' )
            correct_pred += 1
        plt.tick_params( labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both' )
    count+=1

for i in range( len( testing_filelist ) ):
    recogniser( testing_filelist[ i ], train_image_names,proj_data,w )
plt.show( )

print( 'Correct predictions: {}/{} = {}%'.format( correct_pred, num_images, correct_pred / num_images * 100.00 ) )