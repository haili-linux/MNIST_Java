
import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Sigmoid;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.Dense;
import haili.deeplearn.model.layer.SoftmaxLayer;
import haili.deeplearn.utils.ThreadWork;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 *  Fully connect network
 *  全连接神经网络模型
 */
public class FCN_model {

    public static void main(String[] args) throws Exception{

        //导入数据，训练集
        float[][][] train_data = LoadImage.loadMnistData("mnist_data_train.txt");
        float[][] x_train = train_data[0];
        float[][] y_train = train_data[1];

        //导入数据，测试集
        float[][][] test_data = LoadImage.loadMnistData("mnist_data_test.txt");
        float[][] x_test = test_data[0];
        float[][] y_test = test_data[1];

        String modelName = "fc_model.txt";

        Sequential sequential = new Sequential(28 * 28);
        sequential.addLayer(new Dense(100, new LRelu()));
        sequential.addLayer(new Dense(10, new Sigmoid()));
        sequential.addLayer(new SoftmaxLayer());


        sequential = new Sequential(modelName);
        System.out.println(sequential.summary());

        sequential.setDeltaOptimizer(new Adam());
        sequential.setLoss_Function(new CELoss());

        //training 1 epoch
        //sequential.fit(x_train, y_train, 150, 1, 30);

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Command: ");
            String command = sc.next();
            switch (command) {

                case "train":  //训练 training
                {
                    System.out.print("参数输入:epoch batch-size treadN");
                    int epoch, batch, tn;
                    epoch = sc.nextInt();
                    batch = sc.nextInt();
                    tn = sc.nextInt();

                    long t1 = System.currentTimeMillis();
                    sequential.fit(x_train, y_train, batch, epoch, tn);

                    t1 = System.currentTimeMillis() - t1;
                    System.out.println("   time:" + t1);
                    Date d = new Date();
                    SimpleDateFormat sbf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    System.out.println(sbf.format(d));
                }
                    break;

                case "save": {

                    File file = new File(modelName);
                    if (file.exists()) file.delete();

                    sequential.saveInFile(modelName);
                }
                break;

                case "error":
                    System.out.println(sequential.calculateLoss(x_train,y_train));
                    break;

                case "acc"://测试准确率
                    System.out.println("训练集：");
                    testAcc(sequential, x_train, y_train);
                    System.out.println("\n测试集：");
                    testAcc(sequential, x_test, y_test);
                    break;
            }
        }

    }

    public static void loadData(float[][] x, float[][] y, String file_str) {
        File file = new File(file_str);
        File[] files = file.listFiles();

        if (files == null) {
            System.out.println("no found file.");
            return;
        }

        int index = 0;
        for (File value : files) {

            String filename = value.toString();
            int label = Integer.parseInt(filename.substring(filename.length() - 1));
            File[] datalist = value.listFiles();

            if (datalist == null) continue;

            int finalIndex = index;
            ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(datalist.length) {
                @Override
                public void working(int p) {
                    try {
                        x[finalIndex + p] = LoadImage.bmpToRgbList_L(datalist[p].toString());
                        y[finalIndex + p][label] = 1.0f;
                    } catch (Exception exception) {
                        exception.printStackTrace();
                    }
                }
            };

            ThreadWork.start(threadWorker, 16);
            index += datalist.length;
            System.out.println(index);

        }
    }

    //测试准确率
    public static float testAcc(Sequential sequential, float[][] x_train, float[][] y_train) {
        float acc = 0;
        for(int i = 0; i < x_train.length; i++){
            int p = getMaxIndex(sequential.forward(x_train[i]));
            int label = getMaxIndex(y_train[i]);
            if( p == label ) acc++;
        }

        float r = acc / x_train.length;
        System.out.println("正确率：" + acc + "/" + x_train.length + " = " + r);
        return r;
    }

    public static int getMaxIndex(float[] out){
        int max_i = 0;
        double max = 0;
        for (int i = 0; i < out.length; i++) {
            if (out[i] > max) {
                max = out[i];
                max_i = i;
            }
        }
        return max_i;
    }

    public static String getTime() {
        Calendar calendars = Calendar.getInstance();
        calendars.setTimeZone(TimeZone.getTimeZone("GMT+8:00"));
        String year = String.valueOf(calendars.get(Calendar.YEAR));
        String month = String.valueOf(1 + calendars.get(Calendar.MONTH));
        String day = String.valueOf(calendars.get(Calendar.DATE));
        String hour = String.valueOf(calendars.get(Calendar.HOUR));
        String min = String.valueOf(calendars.get(Calendar.MINUTE));
        String second = String.valueOf(calendars.get(Calendar.SECOND));
        //Boolean isAm = calendars.get(Calendar.AM_PM)==1 ? true:false;
        //Boolean is24 = DateFormat.is24HourFormat(getApplication()) ?true:false;
        return year+"年"+month+"月"+day+"日"+hour+"时"+min+"分"+second;
    }


}
