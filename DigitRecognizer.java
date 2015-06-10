import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.Map;
import java.util.Scanner;

import weka.classifiers.functions.SMO;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.KNearestNeighbors;
import net.sf.javaml.classification.evaluation.CrossValidation;
import net.sf.javaml.classification.evaluation.EvaluateDataset;
import net.sf.javaml.classification.evaluation.PerformanceMeasure;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.tools.weka.WekaClassifier;

public class DigitRecognizer
{

	public static void main(String[] args) 
	{
		String trainPath = args[0];
		String testPath = args[1];
		String outputPath = args[2];
		String trainTemp = "trainTemp.csv";
		String testTemp = "testTemp.csv";
		
		try
		{
			/* Pre-process data(remove header lines) */
			removeFirstLine(trainPath, trainTemp);
			String headerLine = removeFirstLine(testPath, testTemp);
			String[] features = headerLine.split(",");
			
			/* Load training data  */
			Dataset trainingData = FileHandler.loadDataset(new File(trainTemp), 0, ",");
			Dataset testData = FileHandler.loadDataset(new File(testTemp), ",");

			/* Build java-ml native classifier */
			Classifier knn = new KNearestNeighbors(7);
			knn.buildClassifier(trainingData);
			
			/* Weka classifier*/
//			SMO smo = new SMO();
//			/* Wrap Weka classifier in bridge */
//			Classifier javamlsmo = new WekaClassifier(smo);
//			javamlsmo.buildClassifier(trainingData);
			
			/* Doing prediction */
			try
			{
				FileWriter fileWriter = new FileWriter(new File(outputPath));  
				BufferedWriter bufferWriter = new BufferedWriter(fileWriter);
				int id = 1;
				bufferWriter.write("ImageId,Label");
				bufferWriter.newLine();
				for (Instance inst : testData) 
				{
				    Object predictedClassValue = knn.classify(inst);				    
				    /* Write each predicted value to csv file */
				    bufferWriter.write((id++) + "," + predictedClassValue.toString());
				    bufferWriter.newLine();
				}
				bufferWriter.close();
			}
			catch (FileNotFoundException e)
			{
				System.out.println("Input file not found.");
				System.exit(0);
			}
			catch(IOException ex)
			{
				System.out.println("Cannot write output file.");
				System.exit(0);
			}

			/* Evaluate classifier performance */
			Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, trainingData);
			for(Object o:pm.keySet())
			    System.out.println(o+": "+pm.get(o).getAccuracy());
			
			/* Classification cross validation */
			CrossValidation cv = new CrossValidation(knn);
			/* Perform cross-validation on the data set */
			Map<Object, PerformanceMeasure> p = cv.crossValidation(trainingData);
			
		}
		catch (IOException ex)
		{
			System.out.println(ex.getMessage());
			System.exit(0);
		}
	}
	
	public static String removeFirstLine(String fileName, String newFileName) throws IOException 
	{  
	    RandomAccessFile raf_read = new RandomAccessFile(fileName, "rw"); 
	    RandomAccessFile raf_write = new RandomAccessFile(newFileName, "rw"); 
	     //Initial write position                                             
	    long writePosition = raf_write.getFilePointer();                            
	    String headerLine = raf_read.readLine();                                                       
	    // Shift the next lines upwards.                                      
	    long readPosition = raf_read.getFilePointer();                             

	    byte[] buff = new byte[1024];                                         
	    int n;                                                                
	   
	    while (-1 != (n = raf_read.read(buff)))
	    {                                  
	        raf_write.seek(writePosition);                                          
	        raf_write.write(buff, 0, n);                                            
	        readPosition += n;                                                
	        writePosition += n;                                               
	        raf_read.seek(readPosition);                                           
	    }                                                                     
	    raf_write.setLength(writePosition);                                         
	    raf_read.close();  
	    raf_write.close();
	    
	    return headerLine;
	}         
}
