import java.io.File;
import java.util.ArrayList;

import haili.deeplearn.function.Function;
import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Relu;
import haili.deeplearn.function.activation.Tanh;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.model.layer.*;

public class QLearning2 {

    Sequential model_main;
    Sequential model_target;
    ArrayList<float[]> train_x;
    ArrayList<float[]> train_y;
    ArrayList<float[]> train_x_buffer;
    ArrayList<float[]> train_y_buffer;



    public static void main(String[] args) {
        QLearning2 qLearning2 = new QLearning2();
        float[][] x = new float[100][];
        float[][] y = new float[100][];
        for(int i = 0; i < x.length; i+=5){
            float[] s = Layer.GaussRandomArrays(qLearning2.model_main.input_dimension - 5);
            for(int j = 0; j < 5; j++){
                float[] act = new float[5];
                act[j] = 1;
                x[i + j] = MatrixUtil.combine(s, act);
                y[i + j] = Layer.GaussRandomArrays(qLearning2.model_main.output_dimension);
            }
        }

        float loss = qLearning2.model_main.calculateLoss(x, y);
        System.out.println(loss);
        qLearning2.model_main.fit(x, y, 10, 20, 4);
        loss = qLearning2.model_main.calculateLoss(x, y);
        System.out.println(loss);
    }

    public QLearning2(){
        train_x = new ArrayList<>();
        train_y = new ArrayList<>();
        train_x_buffer = new ArrayList<>();
        train_y_buffer = new ArrayList<>();

        int w = 10;
        int h = 10;
        int ch = 4;
        Sequential sNet = new Sequential(w, h, w * h * ch);
        sNet.addLayer(new Conv2D(3, 3, 8, 1, new Relu()));
        sNet.addLayer(new Conv2D(3, 3, 16, 1, new Relu()));
        sNet.addLayer(new Dense(128, new LRelu()));

        Sequential qNet = new Sequential(sNet.output_dimension + 5/*act dimension*/);
        ResBlock resBlock_split = new ResBlock(new SplitLayer(sNet.output_dimension, 5), ResBlock.ResConnectType_Concat);
        resBlock_split.addLayer(new Dense(128 - 5));
        resBlock_split.addLayer(new FilterResponseNormalization());
        resBlock_split.addLayer(new ActivationLayer(new LRelu()));
        qNet.addLayer(resBlock_split);

        qNet.addLayer(new Dense(64));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));

        qNet.addLayer(new Dense(32));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));
        qNet.addLayer(new Dense(1));


//        ResBlock resBlock0 = new ResBlock(ResBlock.ResConnectType_Add);
//        resBlock0.addLayer(new Dense(128));
//        resBlock0.addLayer(new FilterResponseNormalization());
//        resBlock0.addLayer(new ActivationLayer(new LRelu()));
//
//        qNet.addLayer(resBlock0);
//
//        qNet.addLayer(new Dense(64));
//        qNet.addLayer(new FilterResponseNormalization());
//        qNet.addLayer(new ActivationLayer(new LRelu()));
//
//        ResBlock resBlock01 = new ResBlock(ResBlock.ResConnectType_Add);
//        resBlock01.addLayer(new Dense(64));
//        resBlock01.addLayer(new FilterResponseNormalization());
//        resBlock01.addLayer(new ActivationLayer(new LRelu()));
//        qNet.addLayer(resBlock01);
//        qNet.addLayer(new Dense(1));


        SplitLayer a_splitLayer = new SplitLayer(w * h * ch, 5);
        ResBlock resBlock = new ResBlock(a_splitLayer, ResBlock.ResConnectType_Concat);
        resBlock.addLayer(new SplitLayer(0, w * h * ch));
        resBlock.addLayer(new Reshape(w, h));
        resBlock.addLayer(sNet);

        model_main = new Sequential(w * h * ch + 5);
//        model_main.addLayer(new Dense(128));
//        model_main.addLayer(new FilterResponseNormalization());
//        model_main.addLayer(new ActivationLayer(new LRelu()));
//        model_main.addLayer(new Dense(64));
//        model_main.addLayer(new FilterResponseNormalization());
//        model_main.addLayer(new ActivationLayer(new LRelu()));
//        model_main.addLayer(new Dense(1));
        model_main.addLayer(resBlock);
        model_main.addLayer(qNet);

        model_main.loss = 999;
        //model_main.setDeltaOptimizer(new Adam());
        model_main.setLearn_rate(1e-4f);
        System.out.println(model_main.summary());

        try {
            updateModel_Target();
        }catch (Exception e){
            model_target = model_main;
        }
    }

    public QLearning2(String fileName){
        train_x = new ArrayList<>();
        train_y = new ArrayList<>();
        train_x_buffer = new ArrayList<>();
        train_y_buffer = new ArrayList<>();
        model_main = new Sequential(fileName);
        model_target = new Sequential(fileName);
        System.out.println(model_main.summary());
    }

    public float[] sample(float[] s, boolean test){
        float[] actList = new float[5];
        double d0 = Math.random();
        if (!test && ( d0 < model_main.loss / 2 || d0 < 0.1)) {
            int index = (int)(Math.random() * actList.length);
            actList[index] = 1;
            return actList;
        }

        float maxr = -999999999;
        int max_index = 0;
        ArrayList<Integer> integersAct = new ArrayList<>();

        for(int i = 0; i < 5; i++){
            float[] act = new float[5];
            act[i] = 1;
            float[] input = MatrixUtil.combine(s, act);
            float pr = model_main.forward(input)[0];

            if(pr > maxr){
                maxr = pr;
                max_index = i;
            } else  if(pr == maxr){
                integersAct.add(i);
            }
//            String act_str = "(";
//            for(int index = input.length - 5; index < input.length; index++){
//                act_str += input[index] + ", ";
//            }

//            System.out.println("play()   act:" +act_str + "   point:"  + pr);
        }

        integersAct.add(max_index);

        if(integersAct.size() > 1){
            int rint = (int)(Math.random() * integersAct.size());
            actList[integersAct.get(rint)] = 1;
        }else{
            actList[max_index] = 1;
        }

        return actList;
    }

    public void updateModel_Target() throws Exception{
        String fileName = Math.random() + "";
        save(fileName);
        model_target = new Sequential(fileName);
        File file = new File(fileName);
        file.delete();
    }


    public float getMaxSorce(float[] s){
        float maxr = -999999999;

        for(int i = 0; i < 5; i++){
            float[] act = new float[5];
            act[i] = 1;
            float[] input = MatrixUtil.combine(s, act);
            float pr = model_target.forward(input)[0];

            if(pr > maxr){
                maxr = pr;
            }
//            System.out.println("play()   act:" + Arrays.toString(act) + "   point:"  + pr);
        }

        return maxr;
    }

    private int dataBufferSize = 256 * 100;

    public static Function tanh = new Tanh();
    public void addData(float[] s, float[] a, float[] r){
        train_x_buffer.add(MatrixUtil.combine(s, a));
        train_y_buffer.add(r);

        int t = train_y_buffer.size();

        if(r[0] > 0){
            if(r[0] >= 1){

                for(int i = 0; i < t; i++){
                    float[] st = train_y_buffer.get(i);
                    st[0] += r[0] * (float)Math.pow(0.9, t - i);//- (t - i - 1) * 0.0001f; //　基礎のポイント - 時間に関する懲罰
                    //st[0] = tanh.f(st[0]);
                }

                train_x.addAll(train_x_buffer);
                train_x_buffer.clear();
                train_y.addAll(train_y_buffer);
                train_y_buffer.clear();
            }
        } else {

            train_x.addAll(train_x_buffer);
            train_x_buffer.clear();
            train_y.addAll(train_y_buffer);
            train_y_buffer.clear();
        }


        while(train_x.size() > dataBufferSize){
            train_x.remove(0);
            train_y.remove(0);
        }
    }


    public void trainning(int epoch, int batch_size, int thread_number){
        if(train_x.size() < batch_size) return;

        float[][] tx = new float[train_x.size()][];
        float[][] ty = new float[train_y.size()][];
        for(int i = 0; i < tx.length; i++){
            tx[i] = train_x.get(i);
            ty[i] = train_y.get(i);
        }

        model_main.fit(tx, ty, batch_size, epoch, thread_number);
        model_main.calculateLoss(tx, ty);
    }



    public void save(String fileName) throws Exception{

        File file = new File(fileName);
        if(file.exists() && file.isFile())
            file.delete();

        model_main.EXPLAIN = GAN.getDataTime();
        model_main.saveInFile(fileName);

    }
}