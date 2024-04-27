
import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.activation.Softmax;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.function.loss.CESLoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

/**
 * Conv NetWork
 * 卷积神经网络模型
 */
public class CnnModel {
    public static void main(String[] args) throws Exception {

        /*
        //训练集
        float[][] x_train = new float[60000][];
        float[][] y_train = new float[60000][10];

        //测试集
        float[][] x_test = new float[10000][];
        float[][] y_test = new float[10000][10];

        String file_train = "dataSet\\train";
        String file_test = "dataSet\\test";

        //直接从图片文件导入数据集,多线程
        LoadImage.loadData(x_train, y_train, file_train);

        //直接从图片文件导入数据集,单线程
        LoadImage.loadDataOneThread(x_test, y_test, file_test);
        */

        //导入数据，训练集
        float[][][] train_data = LoadImage.loadMnistData("mnist_data_train.txt");
        float[][] x_train = train_data[0];
        float[][] y_train = train_data[1];

        //导入数据，测试集
        float[][][] test_data = LoadImage.loadMnistData("mnist_data_test.txt");
        float[][] x_test = test_data[0];
        float[][] y_test = test_data[1];

        //创建新的空模型
        Sequential sequential = new Sequential(28, 28, 28 * 28);
        sequential.addLayer(new Conv2D(3, 3, 8, 1, new LRelu())); //添加一层(3x3),输入通道数和输出通道数都为1的卷积层，激活函数为leaky_relu
        sequential.addLayer(new Pooling2D(2, 2));    // 添加(2x2)的池化层
        sequential.addLayer(new Conv2D(3, 3, 8, 1, new LRelu()));
        sequential.addLayer(new Dropout(0.2));
        sequential.addLayer(new Dense(10, new Sigmoid()));//添加全连接层

        // 打印模型
        System.out.println(sequential.summary());

        // 使用交叉熵损失
        sequential.setLoss_Function(new CELoss());

        //使用Adam梯度优化
        sequential.setDeltaOptimizer(new Adam());

        //训练之前，简单测试模型识别正确率
        System.out.println("训练前-训练集: " + testAcc(sequential, x_train, y_train));
        System.out.println("训练前-测试集: " + testAcc(sequential, x_test, y_test));

        // 训练模型
        sequential.fit(x_train, y_train, 150, 50, 30);

        //训练后，简单测试模型识别正确率
        System.out.println("训练后-训练集: " + testAcc(sequential, x_train, y_train));
        System.out.println("训练后-测试集: " + testAcc(sequential, x_test, y_test));

        // 保存模型
        String fileName = "mnist_Sequential_model.txt"; //保存文件路径
        sequential.saveInFile(fileName);


        //从保存的模型文件导入模型
        Sequential sequential2 = new Sequential(fileName);
        //简单测试模型识别正确率
        System.out.println("导入模型-训练集: " + testAcc(sequential2, x_train, y_train));
        System.out.println("导入模型-测试集: " + testAcc(sequential2, x_test, y_test));


        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Command: ");
            String command = sc.next();
            switch (command) {

                case "learn":  //训练
                    System.out.print("参数输入:epoch batch-size threadN");
                    int n, batch, tn;
                    n = sc.nextInt();
                    batch = sc.nextInt();
                    tn = sc.nextInt();

                    if (batch < 10) {
                        System.out.print("batch小于过小,是否继续Y/N? ");
                        if (!sc.next().equals("Y")) break;
                    }
                    long t1 = System.currentTimeMillis();

                    for (int i = 0; i < n; i++) {
                        sequential.fit(x_train, y_train, batch, 1, tn);
                        System.out.println("   " + sequential.loss);
                    }

                    t1 = System.currentTimeMillis() - t1;

                    System.out.println("   time:" + t1);
                    Date d = new Date();
                    SimpleDateFormat sbf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    System.out.println(sbf.format(d));
                    break;

                case "save":
                    //把模型保存到文件
                    File file = new File("mnist_Sequential_model.txt");
                    if (file.exists()) file.delete();

                    sequential.saveInFile("mnist_Sequential_model.txt");
                    break;

                case "acc"://测试准确率
                    System.out.println("训练集：");
                    testAcc(sequential, x_train, y_train);
                    System.out.println("\n测试集：");
                    testAcc(sequential, x_test, y_test);
                    break;

                case "test":
                    //随机选择10张图片测试
                    for(int i = 0; i < 10; i++){
                        int index = (int) (Math.random()*x_test.length);
                        float[] x = x_test[index];
                        float[] y = y_test[index];

                        float[] out = sequential.forward(x);
                        System.out.println("  truly: " + Arrays.toString(y));
                        System.out.println("predict: " + Arrays.toString(out));
                    }
                    break;
            }

        }
    }

    //测试准确率
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

}
