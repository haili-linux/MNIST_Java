
import haili.deeplearn.BpNetwork;
import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.utils.ThreadWork;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 *  FC network
 */
public class Main_fc {


    public static void main(String[] args) throws Exception{

        //训练集
        float[][] x_train = new float[60000][];
        float[][] y_train = new float[60000][10];

        //测试集
        float[][] x_test = new float[10000][];
        float[][] y_test = new float[10000][10];

        String file_train = "dataSet\\train";
        String file_test = "dataSet\\test";

        //导入数据集
        loadData(x_train, y_train, file_train);
        loadData(x_test, y_test, file_test);


        /*
        int input_vector = 28 * 28;
        int output_vector = 10;

        float learn_rate = 1e-4f;

        int[] hidden = new int[]{ 784, 256, 128, 64, 32, 32, 16 };

        BpNetwork bpNetwork = new BpNetwork(input_vector,output_vector, learn_rate, new LRelu(), hidden);

        for (int i = 0; i < bpNetwork.output_Neuer.length; i++)
            bpNetwork.output_Neuer[i].ACT_function = new Sigmoid();

         */


        BpNetwork bpNetwork = new BpNetwork("mnist_bp.txt");
        bpNetwork.deltaOptimizer = new Adam(bpNetwork.getWandD_number(), 0.9f,0.999f,1e-8f);

        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Command: ");
            String command = sc.next();
            switch (command) {

                case "learnt":
                    System.out.print("参数输入:时间min batch-size 使用线程数");
                    int time2, batcht, tnt;
                    time2 = sc.nextInt();
                    batcht = sc.nextInt();
                    tnt = sc.nextInt();
                    System.out.println("   " + getTime() );
                    bpNetwork.fit_time(x_train, y_train, batcht, tnt,time2);
                    System.out.println("   " + bpNetwork.dError);
                    System.out.println("   " + getTime() );
                    break;


                case "learn":  //学习
                        System.out.print("参数输入:epoch batch-size treadN");
                        int n, batch, tn;
                        n = sc.nextInt();
                        batch = sc.nextInt();
                        tn = sc.nextInt();

                        if(batch < 10) {
                            System.out.print("batch小于过小,是否继续Y/N? ");
                            if(!sc.next().equals("Y")) break;
                        }
                        long t1 = System.currentTimeMillis();

                        for (int i = 0; i < n; i++) {
                            bpNetwork.fit(x_train, y_train, batch, 1, tn);
                            System.out.println("   " + bpNetwork.dError);
                        }

                        t1 = System.currentTimeMillis() - t1;

                        System.out.println("   time:" + t1 );
                        Date d = new Date();
                        SimpleDateFormat sbf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                        System.out.println(sbf.format(d));
                    break;

                case "save":
                    File file = new File("mnist_bp.txt");
                    if (file.exists()) file.delete();

                    bpNetwork.saveInFile("mnist_bp.txt");
                    break;

                case "error":
                    System.out.println(bpNetwork.calculateLoss(x_train,y_train));
                    break;

                case "acc"://测试准确率
                    System.out.println("训练集：");
                    testAcc(bpNetwork, x_train, y_train);
                    System.out.println("\n测试集：");
                    testAcc(bpNetwork, x_test, y_test);
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
    public static float testAcc(BpNetwork bpNetwork, float[][] x_train, float[][] y_train) {
        float acc = 0;
        for(int i = 0; i < x_train.length; i++){
            int p = getMaxIndex(bpNetwork.out_(x_train[i]));
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
