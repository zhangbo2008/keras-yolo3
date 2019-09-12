from keras import backend as K
import tensorflow as tf
grid_shape=[13,13]
grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                [1, grid_shape[1], 1, 1])
grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                [grid_shape[0], 1, 1, 1])
grid = K.concatenate([grid_x, grid_y])  # concatenate 默认-1轴
# with tf.Session() as sess:
#     print(grid.eval(sesion=sess))
with tf.Session() as sess:
    print (sess.run(grid))
    print(1111111111111111111111)
    print (sess.run(grid_y/[1,2]))
    print (sess.run(grid/[1,2]))
