package dqn;

import java.io.File;
import java.time.LocalDateTime;
import java.util.ArrayList;

import haili.deeplearn.function.activation.LRelu;
import haili.deeplearn.function.activation.Relu;
import haili.deeplearn.model.Sequential;
import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.model.layer.*;

public class QNetwork {

    Sequential model_main;
    Sequential model_target;

    public QNetwork(int input_width, int input_height, int input_chanel, int action_dimension){

        /*
        Sequential sNet = new Sequential(input_width, input_height, input_width * input_height * input_chanel);
        sNet.addLayer(new Conv2D(3, 3, 8, 1, new Relu()));
        sNet.addLayer(new Conv2D(3, 3, 16, 1, new Relu()));
        sNet.addLayer(new Dense(128, new LRelu()));


        Sequential qNet = new Sequential(sNet.output_dimension + action_dimension);
        qNet.addLayer(new Dense(128));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));

        qNet.addLayer(new Dense(64));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));

        qNet.addLayer(new Dense(32));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));
        qNet.addLayer(new Dense(1));
        */

        //selfAttention mode
        Sequential sNet = new Sequential(input_width, input_height, input_width * input_height * input_chanel);
        sNet.addLayer(new PositionLayer(4, 2, 25));
        sNet.addLayer(new SelfAttention(6, 32, 32));
        sNet.addLayer(new CombineSequencesLayer(32));
        sNet.addLayer(new FilterResponseNormalization());
        sNet.addLayer(new ActivationLayer(new LRelu()));

        Sequential qNet = new Sequential(sNet.output_dimension + action_dimension);
        qNet.addLayer(new Dense(64));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));

        qNet.addLayer(new Dense(32));
        qNet.addLayer(new FilterResponseNormalization());
        qNet.addLayer(new ActivationLayer(new LRelu()));
        qNet.addLayer(new Dense(1));


        SplitLayer a_splitLayer = new SplitLayer(input_width * input_height * input_chanel, action_dimension);
        ResBlock resBlock = new ResBlock(a_splitLayer, ResBlock.ResConnectType_Concat);
        resBlock.addLayer(new SplitLayer(0, input_width * input_height * input_chanel));
        resBlock.addLayer(new Reshape(input_width, input_height));
        resBlock.addLayer(sNet);

        model_main = new Sequential(input_width * input_height * input_chanel + action_dimension);
        model_main.addLayer(resBlock);
        model_main.addLayer(qNet);

        model_main.loss = 999;
        model_main.setLearn_rate(1e-4f);
        System.out.println(model_main.summary());

        try {
            updateModel_Target();
        }catch (Exception e){
            model_target = model_main;
        }
    }

    public QNetwork(String fileName){
        model_main = new Sequential(fileName);
        model_target = new Sequential(fileName);
        System.out.println(model_main.summary());
    }

    public float[] sample(float[] state, boolean isTest){
        float[] actList = new float[5];
        double d0 = Math.random();
        if (!isTest && ( d0 < 0.1)) {
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
            float[] input = MatrixUtil.combine(state, act);
            float pr = model_main.forward(input)[0];

            if(pr > maxr){
                maxr = pr;
                max_index = i;
            } else  if(pr == maxr){
                integersAct.add(i);
            }
            /*
            String act_str = "(";
            for(int index = input.length - 5; index < input.length; index++){
                act_str += input[index] + ", ";
            }
            System.out.println("play()   act:" +act_str + "   point:"  + pr);
            */
        }

        integersAct.add(max_index);

        if(integersAct.size() > 1){
            int int0 = (int)(Math.random() * integersAct.size());
            actList[integersAct.get(int0)] = 1;
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


    public float getMaxQ(float[] s){
        float maxr = -999999999;

        for(int i = 0; i < 5; i++){
            float[] act = new float[5];
            act[i] = 1;
            float[] input = MatrixUtil.combine(s, act);
            float pr = model_target.forward(input)[0];

            if(pr > maxr){
                maxr = pr;
            }
            //System.out.println("play()   act:" + Arrays.toString(act) + "   point:"  + pr);
        }

        return maxr;
    }

    public void save(String fileName) throws Exception{
        File file = new File(fileName);
        if (file.exists() && file.isFile()) {
            file.delete();
        }

        model_main.EXPLAIN = LocalDateTime.now().toString();
        model_main.saveInFile(fileName);
    }

}