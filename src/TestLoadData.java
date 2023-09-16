public class TestLoadData {
    public static void main(String[] args) throws Exception {
        float[][] x_train = new float['\uea60'][];
        float[][] y_train = new float['\uea60'][10];
        float[][] x_test = new float[10000][];
        float[][] y_test = new float[10000][10];
        String file_train = "dataSet\\train";
        String file_test = "dataSet\\test";
        LoadImage.loadData(x_train, y_train, file_train);
        LoadImage.loadData(x_test, y_test, file_test);
        LoadImage.saveMnistData(x_train, y_train, "mnist_data_train.txt");
        LoadImage.saveMnistData(x_test, y_test, "mnist_data_test.txt");
        System.exit(0);
    }

}
