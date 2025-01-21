import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class MorphologyTest {
    private static final String[] ELEMENT_TYPE = { "Rectangle", "Cross", "Ellipse" };
    private static final String[] MORPH_OP = { "Erosion", "Dilatation" };
    private static final int MAX_KERNEL_SIZE = 21;
    private Mat matImgSrc;
    private Mat matImgDst = new Mat();
    private int elementType = Imgproc.CV_SHAPE_RECT;
    private int kernelSize = 0;
    private boolean doErosion = true;
    private boolean doNegative = false;
    private JFrame frame;
    private JLabel imgLabel;

    //! [constructor]
    public MorphologyTest(String[] args) {
        String imagePath = args.length > 0 ? args[0] : "./test_images/letters.bmp";
        matImgSrc = Imgcodecs.imread(imagePath);
        if (matImgSrc.empty()) {
            System.out.println("Empty image: " + imagePath);
            System.exit(0);
        }

        // Create and set up the window.
        frame = new JFrame("Erosion and dilatation demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(matImgSrc);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);
    }
    //! [constructor]

    //! [components]
    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }

        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));

        JComboBox<String> elementTypeBox = new JComboBox<>(ELEMENT_TYPE);
        elementTypeBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JComboBox<String> cb = (JComboBox<String>)e.getSource();
                if (cb.getSelectedIndex() == 0) {
                    elementType = Imgproc.CV_SHAPE_RECT;
                } else if (cb.getSelectedIndex() == 1) {
                    elementType = Imgproc.CV_SHAPE_CROSS;
                } else if (cb.getSelectedIndex() == 2) {
                    elementType = Imgproc.CV_SHAPE_ELLIPSE;
                }
                update();
            }
        });
        sliderPanel.add(elementTypeBox);

        sliderPanel.add(new JLabel("Kernel size: 2n + 1"));
        JSlider slider = new JSlider(0, MAX_KERNEL_SIZE, 0);
        slider.setMajorTickSpacing(5);
        slider.setMinorTickSpacing(5);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                kernelSize = source.getValue();
                update();
            }
        });
        sliderPanel.add(slider);

        JComboBox<String> morphOpBox = new JComboBox<>(MORPH_OP);
        morphOpBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                JComboBox<String> cb = (JComboBox<String>)e.getSource();
                System.out.println(cb.getItemAt(cb.getSelectedIndex()));
                doErosion = cb.getSelectedIndex() == 0;
                update();
            }
        });
        sliderPanel.add(morphOpBox);

        pane.add(sliderPanel, BorderLayout.PAGE_START);
        imgLabel = new JLabel(new ImageIcon(img));
        pane.add(imgLabel, BorderLayout.CENTER);

        JCheckBox negBox = new JCheckBox("Negative", false);
        negBox.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                JCheckBox box = (JCheckBox)e.getSource();
                doNegative = box.isSelected();
                update();
            }
        });

        sliderPanel.add(negBox);
    }
    //! [components]

    //! [update]
    private void update() {
        //! [kernel]
        Mat element = Imgproc.getStructuringElement(elementType, new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));
        //! [kernel]

        Mat tmp = matImgSrc.clone();
        if(doNegative){
            Core.bitwise_not(tmp,tmp);
        }

        if (doErosion) {
            //! [erosion]
            Imgproc.erode(tmp, matImgDst, element);
            System.out.println("erode");
            //! [erosion]
        } else {
            //! [dilation]
            Imgproc.dilate(tmp, matImgDst, element);
            System.out.println("dilate");
            //! [dilation]
        }
        Image img = HighGui.toBufferedImage(matImgDst);
        imgLabel.setIcon(new ImageIcon(img));
        frame.repaint();
    }
    //! [update]

    //! [main]
    public static void main(String[] args) throws UnsupportedLookAndFeelException, ClassNotFoundException, InstantiationException, IllegalAccessException {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new MorphologyTest(args);
            }
        });
    }
    //! [main]
}
