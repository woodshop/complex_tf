import tensorflow as tf

def cplx_matmul(x, y, transpose_x=False, transpose_y=False, x_is_sparse=False, 
                y_is_sparse=False, name=None):
    a,b = tf.unpack(x)
    if transpose_x:
        a = tf.transpose(a)
        b = tf.transpose(b)
    c,d = tf.unpack(y)
    if transpose_y:
        c = tf.transpose(c)
        d = tf.transpose(d)
    m = tf.shape(a)[0]
    n = tf.shape(c)[1]

    x_ext = tf.concat(0, [tf.concat(1, [a, -b]), 
                          tf.concat(1, [b, a])])
    y_ext = tf.concat(0, [tf.concat(1, [c, -d]), 
                          tf.concat(1, [d, c])])
    z = tf.matmul(x_ext, y_ext, False, False, x_is_sparse, y_is_sparse, 
                  name="CplxMatMul")
    return tf.pack([tf.slice(z, [0, 0], [m, n]), tf.slice(z, [m, 0], [m, n])])


def cplx_div(x, y):
    a,b = tf.unpack(x)
    c,d = tf.unpack(y)
    denom = c**2 + d**2
    return tf.pack([(a*c+b*d)/denom, (b*c-a*d)/denom])


def cplx_tanh(x):
    a,b = tf.unpack(x)
    return cplx_div(tf.pack([tf.tanh(a), tf.tan(b)]), 
                    tf.pack([tf.ones_like(a), tf.tanh(a)*tf.tan(b)]))
