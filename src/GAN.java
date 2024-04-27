import haili.deeplearn.DeltaOptimizer.Adam;
import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.*;
import haili.deeplearn.function.loss.CELoss;
import haili.deeplearn.function.loss.CESLoss;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.model.layer.*;
import haili.deeplearn.model.loss.LossLayer;
import haili.deeplearn.utils.DataSetUtils;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.ProgressBarCmd;
import haili.deeplearn.utils.ThreadWork;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 *  生成对抗模型
 */
public class GAN {


    public static void main(String[] args) throws Exception{

        // 导入原始数据, 训练集 60000张
        float[][][] train_data = LoadImage.loadMnistData("mnist_data_train.txt");

        // 导入原始数据, 测试集 10000张
        float[][][] test_data = LoadImage.loadMnistData("mnist_data_test.txt");
        float[][] x_test = test_data[0];  // 图片
        float[][] y_test = test_data[1];  // 标签0 - 9


        // 原始数据，合并 训练集60000 + 测试集10000，共70000
        int dataNum = train_data[0].length + x_test.length;
        float[][] x_train  = new float[dataNum][];
        float[][] y_train = new float[dataNum][];
        for(int i = 0; i < train_data[0].length; i++){
            x_train[i] = train_data[0][i];
            y_train[i] = train_data[1][i];
        }
        for(int i = 0; i < test_data[0].length; i++){
            x_train[i + train_data[0].length] = x_test[i];
            y_train[i + train_data[1].length] = y_test[i];
        }

//        x_train = x_test;
//        y_train = y_test;


        // 创建判别器
        Sequential predict_mode = new Sequential(28, 28, 28 * 28);
        predict_mode.addLayer(new Conv2D(5, 5, 64, 2, new LRelu()));
        predict_mode.addLayer(new Conv2D(5, 5, 128, 2, new LRelu()));
        predict_mode.addLayer(new Dense(11, new Sigmoid()));
        //Sequential predict_mode =  new Sequential("pre_mode.txt");

        // 打印判别器模型
        System.out.println("判别器: " + predict_mode.summary());



        // 创建生成器，输入 54维随机量 + 10维数字
        Sequential gen_mode = new Sequential(54 + 10);

        gen_mode.addLayer(new Dense( 6 * 6 * 128, new Function(), false));  //全连接层，激活函数默认f(x)=x
        gen_mode.addLayer(new FilterResponseNormalization());   // 归一化
        gen_mode.addLayer(new ActivationLayer(new LRelu()));    // Leaky_ReLU 激活层

        gen_mode.addLayer(new Reshape(6, 6));  // Reshape层， w x h = 6 x 6

        // 反卷积 Conv2DTranspose
        gen_mode.addLayer(new Conv2DTranspose(5, 5, 32, 1, new Function(), false));
        gen_mode.addLayer(new FilterResponseNormalization());
        gen_mode.addLayer(new ActivationLayer(new LRelu()));

        gen_mode.addLayer(new Conv2DTranspose(5, 5, 8,1, new Function(), false));
        gen_mode.addLayer(new FilterResponseNormalization());
        gen_mode.addLayer(new ActivationLayer(new LRelu()));

        gen_mode.addLayer(new Conv2DTranspose(2, 2, 1, 2, new Tanh(), false));

        //Sequential gen_mode = new Sequential("gen_mode.txt");

        // 打印生成器模型
        System.out.println("生成器: " + gen_mode.summary());

        // 定义生成器loss计算方式，使用判别器作为loss_function
        gen_mode.lossLayer = new LossLayer(){
            @Override
            public float[] gradient(float[] y_pre, float[] y_t) {
                // y_pre: 生成图片，28 * 28
                // y_t: 预期生成的图片的真实label
                float[] out_pre = predict_mode.forward(y_pre); // 使用判别器识别生成的图片
                float[] deltas = predict_mode.lossLayer.gradient(out_pre, y_t); // 判别器loss层
                return predict_mode.backward(y_pre, out_pre, deltas)[0];// 使用判别器作为loss，返回生成器loss层梯度
            }

            @Override
            public float loss(float[] y_pre, float[] y_t) {
                return 99999.0f;
            }
        };

        //设置判别器loss_function为交叉熵损失
        predict_mode.setLoss_Function(new CELoss());

        // 设置学习率
        predict_mode.setLearn_rate(1e-4f);
        gen_mode.setLearn_rate(1e-4f);

        //使用 Adam梯度优化
        predict_mode.setDeltaOptimizer(new Adam());
        gen_mode.setDeltaOptimizer(new Adam());

        float gen_rate = 0.1f;

        Scanner scanner = new Scanner(System.in);
        while (true){
            try {
                System.out.print("Enter: ");
                String cmd = scanner.next();
                switch (cmd) {
                    //测试判别器
                    case "testP": {
                        float[] truly = new float[10];
                        float[] gen = new float[10];
                        Random random = new Random();
                        float[][] truly_test_image = new float[10][];
                        for (int i = 0; i < truly.length; i++) {
                            int randomNum = random.nextInt(x_train.length);
                            truly_test_image[i] = add_noise(x_train[randomNum], (float) Math.random() / 20.f); //添加噪声后的
                            float[] pre_t = predict_mode.forward(truly_test_image[i]); //判别结果
                            float[] label_t = y_train[randomNum];

                            int label = 0;
                            for (int j = 0; j < label_t.length; j++) {
                                if (label_t[j] == 1.0f)
                                    label = j;
                                truly[i] += 0.1f * predict_mode.lossLayer.loss_function.f(pre_t[j], label_t[j]);
                            }

                            float[] pre_f = predict_mode.forward(gen_mode.forward(createGenInput(gen_mode, i)));
                            for (int j = 0; j < pre_f.length; j++) {
                                gen[i] += 0.1f * predict_mode.lossLayer.loss_function.f(pre_f[j], 0);
                            }
                            System.out.println(" truly: " + label + "  " + pre_t[label] + "  " + Arrays.toString(pre_t));
                            System.out.println(" gen " + i + "  " + Arrays.toString(pre_f));
                        }
                        System.out.println("  真实数据loss:" + Arrays.toString(truly));
                        System.out.println("  生成数据loss:" + Arrays.toString(gen));
                        System.out.println("  mode loss: "  + predict_mode.loss);
                        System.out.println("  loss:" + loss(gen_mode, predict_mode));
                        LoadImage.showImages(truly_test_image, 28, 28, null);
                    }
                    break;

                    //测试生成器
                    case "testG": {
                        float[][] pre = new float[10][];
                        float[] loss = new float[10];
                        float[][] gen_out = new float[10][];

                        for (int i = 0; i < gen_out.length; i++) {
                            gen_out[i] = gen_mode.forward(createGenInput(gen_mode, i));

                            pre[i] = predict_mode.forward(gen_out[i]); //预测输出
                            float[] label = new float[11];   //目标标签
                            label[i] = 1.0f;

                            for (int j = 0; j < label.length; j++) {
                                loss[i] += 0.1f * predict_mode.lossLayer.loss_function.f(pre[i][j], label[j]);
                            }
                        }

                        System.out.println("  loss: " + Arrays.toString(loss));
                        LoadImage.showImages(gen_out, 28, 28, null);
                        System.out.println(Arrays.toString(gen_mode.forward(new float[gen_mode.input_dimension])));
                        float loss2 = loss(gen_mode, predict_mode);
                        System.out.println("  loss_Value:" + loss2);
                    }
                    break;

                    case "save": {
                        File f1 = new File("pre_mode.txt");
                        File f2 = new File("gen_mode.txt");
                        if(f1.exists()) f1.delete();
                        if(f2.exists()) f2.delete();

                        predict_mode.saveInFile("pre_mode.txt");
                        gen_mode.saveInFile("gen_mode.txt");
                        System.out.println("已保存");
                    }
                    break;

                    //训练，指定训练时间
                    case "learnt": {
                        System.out.print("是否训练:判别器 生成数据 ");
                        int p1, g1;
                        p1 = scanner.nextInt();
                        g1 = scanner.nextInt();

                        System.out.print("训练判别器时生成数据比例:");
                        float f0 = scanner.nextFloat();

                        System.out.print("训练时间:");
                        int train_time = scanner.nextInt(); //此次计划训练时间min
                        long starTime = System.currentTimeMillis();
                        int t = 0; //已经训练时间长

                        int epoch = 0;
                        Frame frame = null;
                        String startTime = getDataTime();
                        System.out.println("  startTime:" + startTime);

                        // 根据原始数据集生成训练数据（添加随机噪声）
                        Map<String, float[][]> dataSet_r = creatTrainData(x_train, y_train);

                        while (t < train_time) {
                            epoch++;

                            // 分割数据集，batch_size=280，batch内的数据为随机
                            ArrayList<float[][]>[] data_pre = DataSetUtils.splitBatch(dataSet_r.get("predict_mode_x"), dataSet_r.get("predict_mode_y"), 280);
                            ArrayList<float[][]> pre_train_x = data_pre[0];
                            ArrayList<float[][]> pre_train_y = data_pre[1];

                            // 创建训练进度条
                            String title = "  epoch: " + epoch + "  ";
                            ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, pre_train_x.size(), 50);
                            System.out.print(progressBarCmd.setProgress(0));

                            // 使用每个batch数据训练
                            for (int bach = 0; bach < pre_train_x.size(); bach++) {
                                // 真实数据
                                float[][] pre_x_bach = pre_train_x.get(bach); //batch_size x 748 真实图片
                                float[][] pre_y_bach = pre_train_y.get(bach); //batch_size x 11 真实标签

                                // 根据比例f0，生成生成图片。限制数量为10的倍数，以平均分配10个标签类
                                int genDataNum = (int)(pre_x_bach.length * f0);
                                genDataNum = 10 - genDataNum % 10 + genDataNum;
                                if(genDataNum < 10) genDataNum = 10;

                                // 生成器生成的数据
                                float[][] pre_x_bach_gen = new float[genDataNum][]; // 生成的图片
                                float[][] pre_y_bach_gen = new float[genDataNum][]; // 真实标签

                                // 生成器训练数据
                                float[][] gen_x_bach = new float[genDataNum][]; //生成图片的随机种子
                                float[][] gen_y_bach = new float[genDataNum][]; //生成图片预期标签

                                // 生成数据数量
                                int var0_int = genDataNum / 10;

                                // 生成训练数据
                                for (int i = 0; i < genDataNum; i++) {
                                    int label = i / var0_int; //预期标签, 10个类型平均

                                    //生成器输入
                                    gen_x_bach[i] = createGenInput(gen_mode, label);
                                    gen_y_bach[i] = new float[11];
                                    gen_y_bach[i][label] = 1f;  //训练生成器用的标签

                                    //生成图像
                                    pre_x_bach_gen[i] = gen_mode.forward(gen_x_bach[i]);
                                    pre_y_bach_gen[i] = new float[11];
                                    pre_y_bach_gen[i][10] = 1f; //训练判别器用的标签
                                }


                                //分别计算判别模型在真实数据和生成数据参数梯度
                                float[][] pre_deltas_X_Yt = null, pre_deltas_G_Yt;
                                if(p1 != 0) {
                                    pre_deltas_X_Yt = predict_mode.gradient(pre_x_bach, pre_y_bach, 28);
                                    pre_deltas_G_Yt = predict_mode.gradient(pre_x_bach_gen, pre_y_bach_gen, 28);
                                    // 把在真实数据和生成数据上的梯度 相加
                                    for (int i = 0; i < pre_deltas_G_Yt.length; i++) {
                                        for (int j = 0; j < pre_deltas_G_Yt[i].length; j++)
                                            pre_deltas_X_Yt[i][j] += gen_rate * pre_deltas_G_Yt[i][j];
                                    }
                                }

                                float[][] gen_layers_deltas = null;
                                //生成器参数梯度
                                if(g1 != 0)
                                    gen_layers_deltas = gen_mode.gradient(gen_x_bach, gen_y_bach, 24);//生成模型梯度

                                //分别更新模型参数
                                if(g1 != 0)
                                    gen_mode.upgradeWeight(gen_layers_deltas);
                                if(p1 != 0)
                                    predict_mode.upgradeWeight(pre_deltas_X_Yt);

                                //每50个batch，显示生成效果
                                if(bach % 2000 == 0) {
                                    float[][] gen_out = testG(gen_mode);
                                    frame = LoadImage.showImages(gen_out, 28, 28, "epoch:" + epoch);
                                }

                                //更新进度
                                System.out.print(progressBarCmd.setProgress(bach + 1));

                                t = (int) (System.currentTimeMillis() - starTime) / 60000;
                                if(t >= train_time){
                                    // 评估模型训练效果
                                    float loss = loss(gen_mode, predict_mode);
                                    String currentTime2 = getDataTime();
                                    System.out.print("  loss_value:" + loss + "  startTime:" + startTime +  "  trainTime:" + train_time +   "min  currentTime:" + currentTime2 + "\n");
                                    break;
                                }
                            }

                            float loss = loss(gen_mode, predict_mode);
                            String currentTime2 = getDataTime();
                            System.out.print("  loss_value:" + loss + "  startTime:" + startTime +  "  trainTime:" + train_time +   "min  currentTime:" + currentTime2 + "\n");
                            t = (int) (System.currentTimeMillis() - starTime) / 60000;
                        }

                    }
                    break;

                    // 训练，指定epoch
                    case "train":{
                        System.out.print("epoch:");
                        int train_epoch = scanner.nextInt(); //此次计划训练时间min


                        int epoch = 0;
                        System.out.println("  startTime:" +  getDataTime());

                        // 根据原始数据集生成训练数据
                        Map<String, float[][]> dataSet_r = creatTrainData(x_train, y_train);

                        LoadImage.showImages(testG(gen_mode), 28, 28, "epoch:" + epoch);

                        while (epoch < train_epoch) {
                            epoch++;
                            // 每5轮更新一次训练集（不同的随机噪声）
                            if(epoch % 5 == 0)
                                dataSet_r = creatTrainData(x_train, y_train);

                            // 分割数据集为
                            ArrayList<float[][]>[] data_pre = DataSetUtils.splitBatch(dataSet_r.get("predict_mode_x"), dataSet_r.get("predict_mode_y"), 280);
                            ArrayList<float[][]> pre_train_x = data_pre[0];
                            ArrayList<float[][]> pre_train_y = data_pre[1];

                            // 显示训练进度条
                            String title = "  epoch: " + epoch + "  ";
                            ProgressBarCmd progressBarCmd = new ProgressBarCmd(title, pre_train_x.size(), 50);
                            System.out.print(progressBarCmd.setProgress(0));

                            // 使用每个batch数据训练
                            for (int bach = 0; bach < pre_train_x.size(); bach++) {
                                // 真实数据
                                float[][] pre_x_bach = pre_train_x.get(bach); // 748 真实图片
                                float[][] pre_y_bach = pre_train_y.get(bach); // 11 真实标签

                                int genDataNum = pre_x_bach.length;

                                // 生成器生成的数据
                                float[][] pre_x_bach_gen = new float[genDataNum][]; // 生成的图片
                                float[][] pre_y_bach_gen = new float[genDataNum][]; // 真实标签

                                // 生成器训练数据
                                float[][] gen_x_bach = new float[genDataNum][]; //生成图片的种子
                                float[][] gen_y_bach = new float[genDataNum][]; //生成图片预期标签

                                int var0_int = genDataNum / 10;

                                // 生成训练数据
                                for (int i = 0; i < genDataNum; i++) {
                                    int label = i / var0_int; //预期标签, 10个类型平均
                                    //生成器输入
                                    gen_x_bach[i] = createGenInput(gen_mode, label);
                                    gen_y_bach[i] = new float[11];
                                    gen_y_bach[i][label] = 1f;

                                    //生成图像
                                    pre_x_bach_gen[i] = gen_mode.forward(gen_x_bach[i]);
                                    pre_y_bach_gen[i] = new float[11];
                                    pre_y_bach_gen[i][10] = 1f;
                                }

                                //分别计算判别模型在真实数据和生成数据参数梯度
                                float[][] pre_deltas_X_Yt = predict_mode.gradient(pre_x_bach, pre_y_bach, 28);
                                float[][] pre_deltas_G_Yt = predict_mode.gradient(pre_x_bach_gen, pre_y_bach_gen, 28);
                                for (int i = 0; i < pre_deltas_G_Yt.length; i++) {
                                    for (int j = 0; j < pre_deltas_G_Yt[i].length; j++)
                                        pre_deltas_X_Yt[i][j] += gen_rate * pre_deltas_G_Yt[i][j]; //梯度相加
                                }

                                //计算生成器梯度
                                float[][] gen_layers_deltas = gen_mode.gradient(gen_x_bach, gen_y_bach, 28);//生成模型梯度

                                // 更新生成器和判别器的参数
                                gen_mode.upgradeWeight(gen_layers_deltas);
                                predict_mode.upgradeWeight(pre_deltas_X_Yt);

                                //更新训练进度条
                                System.out.print(progressBarCmd.setProgress(bach + 1));
                            }

                            // 生成10张图片，标签分别为0-9。显示图片
                            float[][] gen_out = testG(gen_mode);
                            LoadImage.showImages(gen_out, 28, 28, "epoch:" + epoch);

                            float loss = loss(gen_mode, predict_mode);
                            String currentTime2 = getDataTime();
                            System.out.print("  loss_value:" + loss + "  currentTime:" + currentTime2 + "\n");

                            //每5个epoch自动保存模型
                            if(epoch % 5 == 0){
                                File f1 = new File("pre_mode.txt");
                                File f2 = new File("gen_mode.txt");
                                if(f1.exists()) f1.delete();
                                if(f2.exists()) f2.delete();

                                predict_mode.saveInFile("pre_mode.txt");
                                gen_mode.saveInFile("gen_mode.txt");
                            }
                        }
                    }
                    break;

                    //设置模型学习率
                    case "setlr": {
                        System.out.print("pre model:");
                        float f1 = scanner.nextFloat();

                        System.out.print("gen model:");
                        float f2 = scanner.nextFloat();

                        System.out.print("gen_rata:");
                        gen_rate = scanner.nextFloat();

                        predict_mode.setLearn_rate(f1);
                        gen_mode.setLearn_rate(f2);
                    }
                    break;

                    default:
                        break;
                }
            } catch (Exception e){
               e.printStackTrace();
            }
        }

    }

    /**
     * @return 根据m原始nist数据生成训练数据
     */
    private static Map<String, float[][]> creatTrainData(float[][] train_x, float[][] train_y){
        float[][] train_x_pre = new float[train_x.length][];
        float[][] train_y_pre = new float[train_x.length][];

        // 生成数据
        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(train_x_pre.length) {
            @Override
            public void working(int p) {
                try {
                    if(p < train_x.length) {
                        train_x_pre[p] = train_x[p];// add_noise(train_x[p], (float) Math.random() / 20.f);
                        train_y_pre[p] = MatrixUtil.combine(train_y[p], new float[1]);
                    }

                } catch (Exception exception) {
                    exception.printStackTrace();
                    System.exit(0);
                }
            }
        };

        ThreadWork.start(threadWorker, 24);

        HashMap<String, float[][]> map = new HashMap<String, float[][]>();
        map.put("predict_mode_x", train_x_pre);
        map.put("predict_mode_y", train_y_pre);

        return  map;
    }

    public static float[][] testG(Sequential gen_mode){
        float[][] gen_out = new float[10][];
        for (int i = 0; i < gen_out.length; i++) {
            gen_out[i] = gen_mode.forward(createGenInput(gen_mode, i));
        }

        return gen_out;
    }

    public static float loss(Sequential gem, Sequential pre_mode){
        float r = 0;
        for(int i = 0; i < 10; i++){
            float[] out = pre_mode.forward(gem.forward(createGenInput(gem, i)));
            for(int j = 0; j < out.length; j++){
                if(j == i)
                    r += (1 - out[j]) ;
                else
                    r += out[j] ;
            }
        }

        return  r / 10;
    }

    /**
     * 创建生成器的输入
     * @param gen_mode 生成器
     * @param number 预期生成结果的类别，
     * @return 生成器的输入。{ 噪声 + 10维标签 }， 10维标签分别表示数字0-9
     */
    public static float[] createGenInput(Sequential gen_mode, int number){
        if( number > 9)
            number = new Random().nextInt(9);

        float[] label = new float[10];
        label[number] = 1.0f;
        float[] noise = noise(gen_mode.input_dimension - 10);

        return MatrixUtil.combine(noise,  label);
    }

    /**
     * @param rate 噪声系数
     * @return 往x向量添加高斯噪声
     */
    public  static float[] add_noise(float[] x, float rate){
        Random random = new Random();
        float[] r = new float[x.length];
        for(int i = 0; i < x.length; i++){
            r[i] = x[i] +  rate *  (float) random.nextGaussian();
            if(r[i] < -1.0f)
                r[i] = -1.0f;
            else if(r[i] > 1.0f)
                r[i] = 1.0f;
        }
        return r;
    }

    /**
     * @param len 长度
     * @return 生成长度为len的高斯噪声
     */
    public static float[] noise(int len){
        Random random = new Random();
        float[] r = new float[len];
        for(int i = 0; i < len; i++)
            r[i] = (float) random.nextGaussian();

        return r;
    }





    /**
     * @return 获取当前时间
     */
    public static String getDataTime(){
        Date currentDate = new Date();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
       return sdf.format(currentDate);
    }
}
