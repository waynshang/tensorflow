import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#NONE直接輸出值
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#建造數值+noise逼真 X 300行
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


#none無論多少個sample都可以
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
#輸入多少=神經元數量第一層
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#定義輸出層
prediction = add_layer(l1, 10, 1, activation_function=None)
#誤差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

#提升準確率以0.1效率來最小誤差化誤差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.global_variables_initializer()  # 替换成这样就好

sess = tf.Session()
sess.run(init)
#圖片框
fig=plt.figure()
#連續性畫圖
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
#連續畫圖
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
      #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        #紅色寬度5
        lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)















                                  
