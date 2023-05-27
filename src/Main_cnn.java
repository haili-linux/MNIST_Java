import haili.deeplearn.BpNetwork;
import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.LRelu;
import haili.deeplearn.function.MSELoss;
import haili.deeplearn.function.Relu;
import haili.deeplearn.function.Sigmoid;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.Conv2D;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.layer.Pooling2D;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Scanner;

public class Main_cnn {
    public static void main(String[] args) throws Exception {
        //训练集
        float[][] x_train = new float[60000][];
        float[][] y_train = new float[60000][10];

        //测试集
        float[][] x_test = new float[10000][];
        float[][] y_test = new float[10000][10];

        String file_train = "dataSet\\train";
        String file_test = "dataSet\\test";

        //导入数据集
        Main_fc.loadData(x_train, y_train, file_train);
        Main_fc.loadData(x_test, y_test, file_test);

        int input_dimension = x_train[0].length; //28*28
        int width = 28;
        int height = 28;

        //创建模型
//        Sequential sequential = new Sequential(width, height, input_dimension);
//        sequential.addLayer(new Conv2D(3, 3, 1, 1, new LRelu()));
//        sequential.addLayer(new Pooling2D(2, 2));
//        sequential.addLayer(new Conv2D(3, 3, 1, 1, new LRelu()));
//        sequential.addLayer(new Pooling2D(2, 2));
//
//        Sequential fcNetwork = new Sequential();
//        fcNetwork.addLayer(new Dense(32, new LRelu()));
//        fcNetwork.addLayer(new Dense(10, new Sigmoid()));
//
//        sequential.addLayer(fcNetwork);
//
//        sequential.setDeltaOptimizer(new Adam());

        Sequential sequential = new Sequential("mnist_Sequential_model.txt");

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Command: ");
            String command = sc.next();
            switch (command) {

                case "learn":  //学习
                    System.out.print("参数输入:epoch batch-size treadN");
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
                    File file = new File("mnist_Sequential_model.txt");
                    if (file.exists()) file.delete();

                    sequential.saveInFile("mnist_Sequential_model.txt");
                    break;

                case "error":
                    System.out.println(sequential.calculateLoss(x_train, y_train));
                    break;

                case "acc"://测试准确率
                    System.out.println("训练集：");
                    testAcc(sequential, x_train, y_train);
                    System.out.println("\n测试集：");
                    testAcc(sequential, x_test, y_test);
                    break;

                case "test":
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
    public static float testAcc(Sequential model, float[][] x_train, float[][] y_train) {
        float acc = 0;
        for(int i = 0; i < x_train.length; i++){
            int p = getMaxIndex(model.forward(x_train[i]));
            int label = getMaxIndex(y_train[i]);
            if( p == label ) acc++;
        }

        float r = acc / x_train.length;
        System.out.println("正确率：" + acc + "/" + x_train.length + " = " + r);
        return r;
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
