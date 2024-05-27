import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class ImagePredictAttention {
    /**
     * 使用Attention层识别图像
     */

    public static void main(String[] args) throws Exception{
        // 导入原始数据, 训练集 60000张
        float[][][] train_data = LoadImage.loadMnistData("C:\\Tool\\IDEA\\Java Project\\minst手写数字识别\\MNIST_Java\\mnist_data_train.txt");
        float[][] x_train = train_data[0];
        float[][] y_train = train_data[1];

        // 导入原始数据, 测试集 10000张
        float[][][] test_data = LoadImage.loadMnistData("C:\\Tool\\IDEA\\Java Project\\minst手写数字识别\\MNIST_Java\\mnist_data_test.txt");
        float[][] x_test = test_data[0];
        float[][] y_test = test_data[1];


        // 分别把图片分割为16个 7x7 的块
        for(int i = 0; i < x_train.length; i++)
            x_train[i] = reshapeImage(x_train[i], 28, 28, 7, 7);
        for(int i = 0; i < x_test.length; i++)
            x_test[i] = reshapeImage(x_test[i], 28, 28, 7, 7);


        PositionLayer positionLayer = new PositionLayer(63, 1, 16);
        // 创建模型
        Sequential sequential = new Sequential();
        sequential.addLayer(new SlidingWindowLayer(49, new Dense(49, 63, new Function())));
        sequential.addLayer(positionLayer);
        sequential.addLayer(new SelfAttention(64, 64, 64));
        sequential.addLayer(new SlidingWindowLayer(64 ,new Dense(64, 64, new LRelu())));
        sequential.addLayer(new CombineSequencesLayer(64));
        sequential.addLayer(new Dense(64, 10, new Sigmoid()));

        System.out.println(sequential.summary());

        sequential.setLearn_rate(1e-4f);
        sequential.setDeltaOptimizer(new Adam());
        sequential.setLoss_Function(new CELoss());

        //sequential.fit(x_train, y_train, 150, 50, 30);

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Command: ");
            String command = sc.next();
            switch (command) {

                case "train":  //训练
                    System.out.print("参数输入:epoch batch-size threadN");
                    int epoch, batch, tn;
                    epoch = sc.nextInt();
                    batch = sc.nextInt();
                    tn = sc.nextInt();

                    if (batch < 10) {
                        System.out.print("batch小于过小,是否继续Y/N? ");
                        if (!sc.next().equals("Y")) break;
                    }
                    long t1 = System.currentTimeMillis();
                    sequential.fit(x_train, y_train, batch, epoch, tn);
                    t1 = System.currentTimeMillis() - t1;

                    System.out.println("   time:" + t1);
                    Date d = new Date();
                    SimpleDateFormat sbf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    System.out.println(sbf.format(d));
                    break;

                case "save": {
                    String fileName = "mnist_Attention_model.txt";
                    //把模型保存到文件
                    File file = new File(fileName);
                    if (file.exists()) file.delete();

                    sequential.saveInFile(fileName);
                    System.out.println("已保存.");
                } break;

                case "acc"://测试准确率
//                    System.out.println("训练集："  + testAcc(sequential, x_train, y_train));
                    System.out.println("\n测试集："  + testAcc(sequential, x_test, y_test));

                    // 减少输入向量个数，测试模型识别效果
                    float[][] test_data_0_14 = test_data(x_test, 0,14);
                    System.out.println("测试集（0-14块）: " + testAcc(sequential, test_data_0_14, y_test));

                    float[][] test_data_0_12 = test_data(x_test, 0,12);
                    System.out.println("测试集（0-12块）: " + testAcc(sequential, test_data_0_12, y_test));

                    float[][] test_data_0_10 = test_data(x_test, 0,10);
                    System.out.println("测试集（0-10块）: " + testAcc(sequential, test_data_0_10, y_test));

                    float[][] test_data_0_9 = test_data(x_test, 0,9);
                    System.out.println("测试集（0-9块）: " + testAcc(sequential, test_data_0_9, y_test));

                    float[][] test_data_0_8 = test_data(x_test, 0, 8);
                    System.out.println("测试集（0-8块）: " + testAcc(sequential, test_data_0_8, y_test));

                    float[][] test_data_1_9 = test_data(x_test, 1, 9);
                    System.out.println("测试集（1-9块）: " + testAcc(sequential, test_data_1_9, y_test));

                    float[][] test_data_2_10 = test_data(x_test, 2, 10);
                    System.out.println("测试集（2-10块）: " + testAcc(sequential, test_data_2_10, y_test));

                    float[][] test_data_2_15 = test_data(x_test, 2, 15);
                    System.out.println("测试集（2-15块）: " + testAcc(sequential, test_data_2_15, y_test));

                    break;

                case "test": {
                    //随机选择10张图片测试
                    for (int i = 0; i < 10; i++) {
                        int index = (int) (Math.random() * x_test.length);
                        float[] x = x_test[index];
                        float[] y = y_test[index];

                        float[] out = sequential.forward(x);
                        System.out.println("  truly: " + Arrays.toString(y));
                        System.out.println("predict: " + Arrays.toString(out));
                    }
                }
                break;

                default:{
                    for (float[] pi: positionLayer.positionCode)
                        System.out.println(Arrays.toString(pi));
                }
                break;
            }

        }
    }

    //粗略测试准确率
    public static String testAcc(Sequential model, float[][] x_train, float[][] y_train) {
        float acc = 0;
        for(int i = 0; i < x_train.length; i++){
            int p = getMaxIndex(model.forward(x_train[i]));
            int label = getMaxIndex(y_train[i]);
            if( p == label ) acc++;
        }

        return  (100 * acc / x_train.length) + "%";
    }

    static int getMaxIndex(float[] arrays){
        int index = 0;
        float max = arrays[0];
        for (int i = 0; i < arrays.length; i++){
            if(arrays[i] > max){
                max = arrays[i];
                index = i;
            }
        }
        return index;
    }


    /**
     * 把图片分割
     * @param image 图片
     * @param w 图片width
     * @param h 图片h
     * @param split_w 分割成的小块的width
     * @param split_h h
     * @return 分割后的图块平铺后组成的数组
     */
    private static float[] reshapeImage(float[] image, int w, int h, int split_w, int split_h){
        int sw_n = w / split_w;
        int sh_n = h / split_h;
        if(w < split_w || h < split_h || w % split_w != 0 || h % split_h != 0) {
            System.out.println("不可分割.");
            return null;
        }
        float[] reshapeImage = new float[image.length];

        int sn = sw_n * sh_n;
        int s_len = split_w * split_h;

        //float[][] sImage = new float[sn][s_len];
        for(int i = 0; i < sn; i++){
            int ds = i * s_len;
            int sx = i % sw_n;
            int sy = i / sw_n;

            for(int j = 0; j < s_len; j++){
                int x_ = j % split_w;
                int y_ = j / split_w;

                int x = sx * split_w + x_;
                int y = sy * split_h + y_;
                int index_x_y = x * w + y;
                reshapeImage[ds + j] = image[index_x_y];// +  (i / 160f); // 额外加上位置编码
                //sImage[i][j] = image[index_x_y];
            }
        }
        //LoadImage.showImages(sImage, split_w, split_h, 4, 4,null);
        //LoadImage.showImages(new float[][]{image}, w, h, null);
        return  reshapeImage;
    }

    private static float[][] test_data(float[][] data, int start, int end){
        float[][] r = new float[data.length][];
        for(int i = 0; i < data.length; i++){
            float[] di = new float[(end - start) * 7 * 7];
            int start_index = start * 7 * 7;
            System.arraycopy(data[i], start_index , di, 0, di.length);
            r[i] = di;
        }

        return r;
    }
}
