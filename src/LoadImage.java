



import haili.deeplearn.BpNetwork;
import haili.deeplearn.utils.ThreadWork;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public class LoadImage {

    /**
     * bmp图片像素转数组
     * @param imgsrc bmp图片
     * @return 。
     * @throws Exception null
     */
    public static float[] bmpToRgbList_L(BufferedImage imgsrc) {
        float[] rgb = new float[imgsrc.getWidth() * imgsrc.getHeight()];

        //创建一个灰度模式的图片
        BufferedImage back = new BufferedImage(imgsrc.getWidth(), imgsrc.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        int width = imgsrc.getWidth();
        int height = imgsrc.getHeight();

        int index = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                back.setRGB(i, j, imgsrc.getRGB(i, j));
                rgb[index] = (back.getRGB(i,j) &  0xff) / 255.0f;
                index++;
            }
        }
        return rgb;
    }

    public static float[] bmpToRgbList_L(String filename) throws Exception {
        BufferedImage imgsrc = ImageIO.read(new File(filename));
        return  bmpToRgbList_L(imgsrc);
    }

    public static void loadDataOneThread(float[][] x, float[][] y, String file_str) throws Exception{
        File file = new File(file_str);
        File[] files = file.listFiles();

        if (files == null)
            throw new Exception("no found file.");

        int index = 0;
        for(File value: files) {
            String filename = value.toString();
            int label = Integer.parseInt(filename.substring(filename.length() - 1));
            File[] images = value.listFiles();

            for (File image : images) {
                x[index] = bmpToRgbList_L(image.toString());
                y[index][label] = 1;
                index++;
            }
            System.out.println("读取mnist数据: " + index + "张.");
        }
    }

    public static void loadData(float[][] x, float[][] y, String file_str) throws Exception{
        File file = new File(file_str);
        File[] files = file.listFiles();

        if (files == null)
            throw new Exception("no found file.");

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
            System.out.println("读取mnist数据: " + index + "张.");
        }
    }

    public static void saveMnistData(float[][] x, float[][] label, String fileName) throws Exception{

        File f = new File(fileName);
        if(f.exists() && f.isFile()){
            throw new Exception("fileName is exists.");
        } else {
            boolean b = f.createNewFile();
            if(!b) {
                throw new Exception("create file fail.");
            }

            FileWriter fw = new FileWriter(f, true);
            PrintWriter pw = new PrintWriter(fw);

            pw.println("Image Number:" + x.length);
            pw.println("x dimension:" + x[0].length);
            pw.println("y dimension:" + label[0].length);
            for(int i = 0; i < x.length; i++)
                pw.println(Arrays.toString(x[i]).replace(" ","") + ":" + Arrays.toString(label[i]).replace(" ","") );

            pw.flush();
            fw.flush();
            pw.close();
            fw.close();
        }
    }

    public static float[][][] loadMnistData(String fileName) throws Exception{
        File file = new File(fileName);

        FileReader fileReader = null;
        fileReader = new FileReader(file);
        BufferedReader in = new BufferedReader(fileReader);

        String line = in.readLine();
        int len = Integer.parseInt( line.substring(line.lastIndexOf(":") +1) );
        line = in.readLine();
        int x_dimension = Integer.parseInt( line.substring(line.lastIndexOf(":") +1));
        line = in.readLine();
        int label_dimension = Integer.parseInt( line.substring(line.lastIndexOf(":") +1) );

        float[][] x = new float[len][x_dimension];
        float[][] label = new float[len][label_dimension];

        for(int i = 0; i < len; i++){
            line = in.readLine();
            int index = line.indexOf(":");
            String x_str = line.substring(1, index - 1);
            String label_str = line.substring(index + 2, line.length() - 1);
            String[] x_ = x_str.split(",");
            String[] y_ = label_str.split(",");

            for(int j = 0; j < x_dimension; j++) x[i][j] = Float.parseFloat(x_[j]);
            for(int j = 0; j < label_dimension; j++) label[i][j] = Float.parseFloat(y_[j]);
        }

        in.close();
        fileReader.close();

        return new float[][][]{x, label};
    }

}
