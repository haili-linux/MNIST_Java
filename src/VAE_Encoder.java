
import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;
import haili.deeplearn.model.loss.LossLayer;
import haili.deeplearn.utils.MatrixUtil;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;


public class VAE_Encoder {

    public static void main(String[] args) throws Exception{
        // 导入原始数据, 训练集 60000张
        float[][][] train_data = LoadImage.loadMnistData("mnist_data_train.txt");

        // 导入原始数据, 测试集 10000张
        float[][][] test_data = LoadImage.loadMnistData("mnist_data_test.txt");

        int dataNum = train_data[0].length + test_data[0].length;
        //原始数据，训练集60000 + 测试集10000，共70000
        float[][] x_train  = new float[dataNum][];
        System.arraycopy(train_data[0], 0, x_train, 0, train_data[0].length);
        System.arraycopy(test_data[0], 0, x_train, train_data[0].length, test_data[0].length);


        int latent_dim = 2;
        //Sequential encoder_model = new Sequential("vae_encoder_model.txt");
        // 创建编码器
        Sequential encoder_model = new Sequential(28, 28, 28 * 28);
        encoder_model.addLayer(new Conv2D(5, 5, 32, 2, new LRelu()));
        encoder_model.addLayer(new Conv2D(3, 3, 64, 2, new LRelu()));
        encoder_model.addLayer(new Dense(latent_dim + latent_dim));

        // 自定义层，添加gauss noise
        Layer addGaussNoiseLayer = new Layer(){
            @Override
            public void init(int input_width, int input_height, int input_Dimension) {
                this.input_width = input_width;
                this.input_height = input_height;
                this.input_dimension = input_Dimension;
                this.output_width = 3 * latent_dim;
                this.output_height = 1;
                this.output_dimension = 3 * latent_dim;
            }

            @Override
            public float[] forward(float[] inputs) {
                int len = inputs.length / 2;
                float[] outputs = new float[inputs.length + len];
                float[] gaussNoise = GaussRandomArrays(len); //正态分布随机数

                System.arraycopy(inputs, 0, outputs, 0, inputs.length);
                System.arraycopy(gaussNoise, 0, outputs,  inputs.length, len);

                return outputs; // outputs = { [mean], [logVar], [noise] };
            }
        };

        // 自定义reparameterize层
        Layer reparameterizeLayer = new Layer(){
            @Override
            public void init(int input_width, int input_height, int input_Dimension) {
                this.input_width = input_width;
                this.input_height = input_height;
                this.input_dimension = input_Dimension;
                this.output_width = latent_dim;
                this.output_height = 1;
                this.output_dimension = latent_dim;
            }

            /**
             * @param inputs = {[mean0, mean1, ..., mean_N], [logVar0, logVar1, ...., logVarN], [noise0, noise1,...]}
             * @return mean + exp(logVar0) * normal_distribution
             */
            @Override
            public float[] forward(float[] inputs) {
                int len = inputs.length / 3;

                float[] outputs = new float[len];

                for(int i = 0; i < len; i++){
                    int index_logVar0 = i + len;
                    int index_noise = index_logVar0 + len;
                    float mean = inputs[i];
                    float logVar = inputs[index_logVar0];
                    float noise = inputs[index_noise];
                    outputs[i] = (float) Math.exp(logVar) * noise + mean;
                }
                return outputs;
            }

            @Override
            public float[][] backward(float[] inputs, float[] output, float[] deltas) {
                int len = inputs.length / 3;
                float[] mean_logVar_deltas = new float[len * 2];
                for(int i  = 0; i < len; i++){
                    int index_logVar0 = i + len;
                    //int index_noise = index_logVar0 + len;

                    float mean = inputs[i];
                    //float logVar = inputs[index_logVar0];
                    //float noise = inputs[index_noise];

                    mean_logVar_deltas[i] = deltas[i];
                    //mean_logVar_deltas[index] = (float) Math.exp(inputs[index]) * inputs[index + len];
                    mean_logVar_deltas[index_logVar0] = output[i] - mean;
                }

                return new float[][]{mean_logVar_deltas, new float[0]};
            }
        };

        //Sequential decoder_model = new Sequential("vae_decoder_model.txt");
        // 创建解码器
        Sequential decoder_model = new Sequential(latent_dim);
        decoder_model.addLayer(new Dense(6 * 6 * 64, new Function(), false));
        decoder_model.addLayer(new FilterResponseNormalization());
        decoder_model.addLayer(new ActivationLayer(new LRelu()));
        decoder_model.addLayer(new Reshape(6 ,6));
        decoder_model.addLayer(new Conv2DTranspose(5,5,32,1, new Function(), false));
        decoder_model.addLayer(new FilterResponseNormalization());
        decoder_model.addLayer(new ActivationLayer(new LRelu()));
        decoder_model.addLayer(new Conv2DTranspose(5,5,16,1, new Function(), false));
        decoder_model.addLayer(new FilterResponseNormalization());
        decoder_model.addLayer(new ActivationLayer(new LRelu()));
        decoder_model.addLayer(new Conv2DTranspose(2,2,1,2, new Tanh(), false));

        // 创建残差块，将encoder输出以拼接的方式传递到输出，以便在loss里 约束mean和logVar
        ResBlock resBlock = new ResBlock(latent_dim + latent_dim, ResBlock.ResConnectType_Concat);
        resBlock.addLayer(addGaussNoiseLayer);
        resBlock.addLayer(reparameterizeLayer);
        resBlock.addLayer(decoder_model);

        // 创建sequential容器，连接encoder和decoder
        Sequential sequential = new Sequential();
        sequential.addLayer(encoder_model);
        sequential.addLayer(resBlock);
        // outputs = {[image0~728], [mean0,m1, .., mean_N], [logVar0,..., logVarN] }
        System.out.println(sequential.summary()); // 打印模型

        // 自定义模型损失loss
        sequential.lossLayer = new LossLayer(){
            /** 定义loss梯度
             * 分别 Minimize reconstruction error、 Minimize loss_ = exp(logVar) - (1 - logVar) + mean * mean
             * @param y_pre = outputs = {[image 0 ~ 28*28], [mean0,m1, ..., mean_N], [logVar0, ..., logVarN] }
             * @param y_t image x = [28 * 28]
             * @return grads
             */
            @Override
            public float[] gradient(float[] y_pre, float[] y_t) {
                float[] grads = new float[y_pre.length];
                // Minimize reconstruction error
                for(int i = 0 ; i < decoder_model.output_dimension; i++)
                    grads[i] = loss_function.f_derivative(y_pre[i], y_t[i]);

                //Minimize loss_ = exp(logVar) - (1 - logVar) + mean * mean
                for(int i = 0 ; i < latent_dim; i++){
                    int index_mean = encoder_model.output_dimension + i;
                    int index_logVar = index_mean + latent_dim;

                    //D loss_/D mean = mean * 2 / latent_dim
                    float mean_i = y_pre[index_mean];
                    grads[index_mean] = mean_i * 2.0f;

                    //D loss_/D logVar
                    float logVar_i = y_pre[index_logVar];
                    grads[index_logVar] = (float) Math.exp(logVar_i) - 1.0f;
                }

                return grads;
            }

            @Override
            public float[] loss_arrays(float[] y_pre, float[] y_t) {
                float[] loss = new float[y_pre.length];

                // loss, y_pre close to x
                for(int i = 0; i < y_t.length; i++)
                    loss[i] = loss_function.f(y_pre[i], y_t[i]);

                // loss_ = exp(logVar) - (1 - logVar) + mean * mean
                for(int i = 0; i < latent_dim; i++){
                    int index_mean = encoder_model.output_dimension + i;
                    int index_logVar = index_mean + latent_dim;
                    float mean = y_pre[index_mean];
                    float logVar = y_pre[index_logVar];

                    loss[index_mean] = (float) Math.exp(logVar) - (1.0f + logVar) + mean * mean;
                }

                return loss;
            }

            @Override
            public float loss(float[] y_pre, float[] y_t) {
                return MatrixUtil.sum(loss_arrays(y_pre, y_t)) / y_pre.length;
            }
        };
        // 使用Adam梯度优化
        sequential.setDeltaOptimizer(new Adam());
        // 训练
        // sequential.fit(x_train, x_train, 256, 10, 30);

        Scanner scanner = new Scanner(System.in);
        while (true) {
            try {
                System.out.print("Enter: ");
                String cmd = scanner.next();
                switch (cmd) {
                    case "test": {
                        //随机挑选10个数据测试
                        float[][] test_image = new float[10][];
                        float[][] test_image_out = new float[10][];

                        Random random = new Random();
                        for (int i = 0; i < test_image.length; i++) {
                            int randomNum = random.nextInt(x_train.length);
                            test_image[i] = x_train[randomNum];

                            float[] code = encoder_model.forward(test_image[i]);
                            code = addGaussNoiseLayer.forward(code);
                            code = reparameterizeLayer.forward(code);

                            System.out.println("  " + i + " code:" + Arrays.toString(code));
                            test_image_out[i] = decoder_model.forward(code);
                        }

                        System.out.println(" loss:" + sequential.calculateLoss(test_image, test_image_out));

                        // 显示图片
                        LoadImage.showImages(test_image, 28, 28, "原始数据");
                        LoadImage.showImages(test_image_out, 28, 28, "编码-还原");

                        System.out.print("inputs x y: ");
                        float x = scanner.nextFloat();
                        float y = scanner.nextFloat();
                        //显示隐空间中数字的二维流形
                        show_latent_images(decoder_model, x, y);
                    }
                    break;

                    //保存
                    case "save": {
                        encoder_model.saveInFile("vae_encoder_model.txt");
                        decoder_model.saveInFile("vae_decoder_model.txt");
                        System.out.println("已保存");
                    }
                    break;

                    //训练
                    case "train": {
                        System.out.print("epoch:");
                        int epoch = scanner.nextInt();

                        System.out.print("batch-size:");
                        int batch_size = scanner.nextInt();

                        // 训练模型
                        sequential.fit(x_train, x_train, batch_size, epoch, 30);
                    }
                    break;

                    default:
                        break;
                }
            } catch (Exception e) {
                System.out.println(" Exception :" + e.getMessage());
            }
        }
    }


    // 显示隐空间中数字的二维流形
    public static void show_latent_images(Sequential decoder_model, float x_, float y_){
        JFrame frame = new JFrame();
        frame.setTitle("隐空间中数字的二维流形");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setLayout(new GridLayout(30, 30)); // 2 rows, 5 columns

        float x = -x_, y = -y_;
        float dx = -x * 2 / 30;
        float dy =  -y * 2 / 30;
        for (int i = 0; i < 30; i++) {
            for(int j = 0; j < 30; j++) {
                float[] code = {x + i * dx, y + j * dy};  //  -3 <= c <= 3
                float[] image = decoder_model.forward(code);
                BufferedImage img = LoadImage.arraysToImage(image, 28, 28);
                ImageIcon icon = new ImageIcon(img);
                JLabel label = new JLabel(icon);
                frame.add(label);
            }
        }

        frame.pack(); // 自动调整窗口大小以适应图片
        frame.setSize(28* 30 + 200, 28 * 30 + 170);
        frame.setVisible(true);
    }

}
