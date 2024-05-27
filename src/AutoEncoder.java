import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

import java.io.File;
import java.util.*;


public class AutoEncoder {
    public static void main(String[] args) throws Exception {

        // 导入原始数据, 训练集 60000张
        float[][][] train_data = LoadImage.loadMnistData("mnist_data_train.txt");

        // 导入原始数据, 测试集 10000张
        float[][][] test_data = LoadImage.loadMnistData("mnist_data_test.txt");

        int dataNum = train_data[0].length + test_data[0].length;
        // 原始数据，训练集60000 + 测试集10000，共70000
        float[][] x_train  = new float[dataNum][];
        System.arraycopy(train_data[0], 0, x_train, 0, train_data[0].length);
        System.arraycopy(test_data[0], 0, x_train, train_data[0].length, test_data[0].length);

        // AutoEncoder
        // 创建编码器
        Sequential encoder_model = new Sequential(28, 28, 28 * 28);
        encoder_model.addLayer(new Conv2D(5, 5, 16, 2, new LRelu()));
        encoder_model.addLayer(new Conv2D(3, 3, 16, 1, new LRelu()));
        encoder_model.addLayer(new Dense(32, new Tanh()));
        //Sequential encoder_model = new Sequential("encoder_model.txt");

        // 创建解码器
        Sequential decoder_model = new Sequential(32, 1, 32);
        decoder_model.addLayer(new Dense(7 * 7 * 64 , new Function(), false));
        decoder_model.addLayer(new FilterResponseNormalization());
        decoder_model.addLayer(new ActivationLayer(new LRelu()));
        decoder_model.addLayer(new Reshape(7 , 7));
        decoder_model.addLayer(new Conv2DTranspose(2,2,16,2, new Function(), false));
        decoder_model.addLayer(new FilterResponseNormalization());
        decoder_model.addLayer(new ActivationLayer(new LRelu()));
        decoder_model.addLayer(new Conv2DTranspose(2,2,1,2, new Tanh(), false));
        // output = 28 * 28
        //Sequential decoder_model = new Sequential("decoder_model.txt");

        System.out.println("Encoder: " + encoder_model.summary());
        System.out.println("Decoder: " + decoder_model.summary());

        // 创建sequential连接编码器和解码器
        Sequential sequential = new Sequential();
        sequential.addLayer(encoder_model);
        sequential.addLayer(decoder_model);

        sequential.setDeltaOptimizer(new Adam());
        sequential.setLearn_rate(1e-4f);

        Scanner scanner = new Scanner(System.in);
        while (true){
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
                            System.out.println("  " + i + " code:" + Arrays.toString(code));
                            test_image_out[i] = decoder_model.forward(code);
                        }

                        // 显示图片
                        LoadImage.showImages(test_image, 28, 28, "原始数据");
                        LoadImage.showImages(test_image_out, 28, 28, "编码-还原");
                    }
                    break;

                    //保存
                    case "save": {
                        String fileName_encoder = "encoder_model.txt";
                        String fileName_decoder = "decoder_model.txt";

                        if(new File(fileName_decoder).exists())
                            new File(fileName_decoder).delete();
                        if(new File(fileName_encoder).exists())
                            new File(fileName_encoder).delete();

                        encoder_model.saveInFile(fileName_encoder);
                        decoder_model.saveInFile(fileName_decoder);
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
            } catch (Exception e){
                System.out.println(" Exception :" + e.getMessage());
            }
        }
    }

}
