package dqn;

import java.util.ArrayList;

import haili.deeplearn.utils.MatrixUtil;
import haili.deeplearn.utils.ThreadWork;


public class QNetworkTrainer {
    static int WIDTH = 10;
    static int HEIGHT = 10;
    static final String modeName = "snakeGame10x10.txt";
    static SnakeGame snakeGame = new SnakeGame(WIDTH, HEIGHT, true);
    //static QNetwork qNetwork = new QNetwork(WIDTH, HEIGHT, 3, SnakeGame.acts.length);
    static QNetwork qNetwork = new QNetwork(modeName);


    static boolean Test_model = true; //training:false
    static float Beta = 0.9f;
    static int multi_step = 1;
    static int trainDataLen =  1024 * 64 * 2;
    static int dataLen = trainDataLen * multi_step + multi_step;
    static int update_Target_Model_Step = 1024 * 8 * 4;

    static int save_model_time = 10 * 60 * 1000; //10min

    static TrainData trainData = new TrainData(dataLen);

    public static void main(String[] args) throws Exception{

        System.out.println("loss: " + qNetwork.model_main.loss);
        snakeGame.initGame();

        long time = System.currentTimeMillis();
        int step = 0;
        while (true) {
            if (!snakeGame.start) {
                Thread.sleep(10);
                continue;
            }

            snakeGame.VIEW = Test_model;
            snakeGame.render();

            float[] state = snakeGame.state();

            float[] act = qNetwork.sample(state, Test_model);

            snakeGame.doAction(act);
            snakeGame.update();

            if (!Test_model) {
                trainData.add(state, act, snakeGame.reward(), snakeGame.gameOver ? 0 : 1);
            }

            if (!Test_model && multi_step > 1) {
                for (int i = 1; i < multi_step; i++) {
                    if (snakeGame.gameOver) {
                        trainData.add(new float[0], new float[0], 0, 0);

                    } else {
                        snakeGame.render();
                        float[] state_ti = snakeGame.state();
                        float[] act_ti = qNetwork.sample(state, true);
                        snakeGame.doAction(act_ti);
                        snakeGame.update();

                        trainData.add(state_ti, act_ti, snakeGame.reward(), snakeGame.gameOver ? 0:1);
                    }
                }
            }

            trainData.removeData();

            if(snakeGame.gameOver){
                snakeGame.initGame();
            }

            if (!Test_model) {
                long nowtime = System.currentTimeMillis();
                if (nowtime - time > save_model_time) {
                    qNetwork.save(modeName);
                    time = nowtime;
                }

                step ++;
                if (trainData.size() == dataLen && step >= update_Target_Model_Step) {
                    qNetwork.updateModel_Target();
                    step = 0;
                    training(trainData);
                }

            } else {
                Thread.sleep(200);
            }
        }

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
    public static float[] reshapeImage(float[] image, int w, int h, int split_w, int split_h){
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


    private static void training(TrainData trainData) {
        ArrayList<float[]> statesList = trainData.getStatesList();
        ArrayList<float[]> actList = trainData.getActList();
        ArrayList<Float> rewardList = trainData.getRewardList();
        ArrayList<Integer> gameOverList = trainData.getGameOverList();

    
        float[][] train_x = new float[trainDataLen][];
        float[][] train_y = new float[trainDataLen][];

        ThreadWork.ThreadWorker threadWorker = new ThreadWork.ThreadWorker(trainDataLen) {
            @Override
            public void working(int index) {
                int index_base = index * multi_step;
                train_x[index] = MatrixUtil.combine(statesList.get(index_base), actList.get(index_base));
                if (train_x[index].length < 1) {
                    System.out.println();
                }

                train_y[index] = new float[1];
                float beta = 1f;
                for (int i = 0; i < multi_step; i++) {
                    int index_d = index_base + i;
                    if (gameOverList.get(index_d) == 1) {
                        train_y[index][0] += beta * rewardList.get(index_d);
                    } else {
                        train_y[index][0] += beta * rewardList.get(index_d);
                        return;
                    }
                    beta *= Beta;
                }

                train_y[index][0] += beta * qNetwork.getMaxQ(statesList.get( index_base + multi_step ));
            }

        };
        ThreadWork.start(threadWorker, 24);

        System.out.println("loss: " + qNetwork.model_main.calculateLoss(train_x, train_y));
        qNetwork.model_main.fit(train_x, train_y, 512, 5, 24);
    }


    private static class TrainData {
        int dataLen;

        ArrayList<float[]> statesList;
        ArrayList<float[]> actList;
        ArrayList<Float> rewardList;
        ArrayList<Integer> gameOverList;

        public TrainData(int dataLen) {
            this.dataLen = dataLen;
            statesList = new ArrayList<>(dataLen);
            actList = new ArrayList<>(dataLen);
            rewardList = new ArrayList<>(dataLen);
            gameOverList = new ArrayList<>(dataLen);
        }

        public void add(float[] state, float[] action, float reward, int gameOver) {
            statesList.add(state);
            actList.add(action);
            rewardList.add(reward);
            gameOverList.add(gameOver);
        }

        public void removeData() {
            while (statesList.size() > dataLen) {
                for (int i = 0; i < multi_step; i++) {
                    statesList.remove(0);
                    actList.remove(0);
                    rewardList.remove(0);
                    gameOverList.remove(0);
                }
            }
        }

        public int size() {
            return statesList.size();
        }

        public ArrayList<float[]> getActList() {
            return actList;
        }

        public ArrayList<Integer> getGameOverList() {
            return gameOverList;
        }

        public ArrayList<Float> getRewardList() {
            return rewardList;
        }

        public int getDataLen() {
            return dataLen;
        }

        public ArrayList<float[]> getStatesList() {
            return statesList;
        }
    }

}
