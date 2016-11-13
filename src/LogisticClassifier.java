import java.util.Random;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Vector;


public class LogisticClassifier {
	private double[] w1;
	private double b1;
	private double alpha = 1.2;
	private double lambda = 0.02;

	
	// the model parameters
	public int numOfFeature; // number of patterns/features
	public int numOfData; // number of training data/graphs
	public Vector< Vector<Integer> > allData; // all the input data excluding labels;
	public Vector<Integer> allLabel; // all the labels;
	
	public Vector< Vector<Integer> > block;
	private Vector< Vector<Integer> > trainData;
	private Vector<Integer> trainLabel;
	private Vector< Vector<Integer> > testData;
	private Vector<Integer> testLabel;
	

	public LogisticClassifier() {
		// 10-fold cross validation initialization
		block = new Vector< Vector<Integer> >();
		for(int i = 0; i < 10; i++) {
			Vector<Integer> e = new Vector<Integer>();
			block.add(e);
		}
		allData = new Vector< Vector<Integer> >();
		allLabel = new Vector<Integer>();
		trainData = new Vector< Vector<Integer> >();
		trainLabel = new Vector<Integer>();
		testData = new Vector< Vector<Integer> >();
		testLabel = new Vector<Integer>();
	}
	
	public void shuffle() {
		// Uniformly shuffle columns into blocks
		Random rng = new Random();
		for(int i = 0; i < numOfData; i++) {
			int bid = rng.nextInt(10);
			block.elementAt(bid).add(i);
		}
	}
	
	// k, fold number, 0 - 9;
	// matrix includes all the data
	public void makeTrainingData(int k) {
		// Prepare training data
		trainData.clear();
		trainLabel.clear();
		testData.clear();
		testLabel.clear();
		for(int l = 0; l < 10; l++) {
			if(l != k) {
				Vector<Integer> v = block.elementAt(l);
				for(int nj = 0; nj < v.size(); nj++) {
					int j = v.elementAt(nj);
					trainData.add(allData.elementAt(j));
					trainLabel.add(allLabel.elementAt(j));
				}
			}
		}
		Vector<Integer> t = block.elementAt(k);
		for(int nj = 0; nj < t.size(); nj++) {
			int j = t.elementAt(nj);
			testData.add(allData.elementAt(j));
			testLabel.add(allLabel.elementAt(j));
		}
	}
	
	public void loadData(String filename) {
		allData.clear();
		allLabel.clear();
		try {
			//works
			String line;
			FileReader file = new FileReader(filename);
			BufferedReader b = new BufferedReader(file);
			int iterations = 0;
			while((line = b.readLine()) != null) {
				String[] stringLine = line.split(" ");
				int[] line1 = new int[stringLine.length];
				// 1st line
				if (iterations == 0) {
					for (int i = 0; i < stringLine.length; i++) {
						line1[i] = Integer.valueOf(stringLine[i]);
					}
					numOfData = line1[0];
					numOfFeature = line1[1];
					iterations++;
				}
				else if(iterations%2 == 0) {
					Vector<Integer> nv = new Vector<Integer>();
					for (int i = 0; i < stringLine.length; i++) {
						line1[i] = Integer.valueOf(stringLine[i]);
					}
					if (line1[0] != -1) {
						for (int i = 0; i < line1.length; i++) {
							nv.add(line1[i]);;
						}
					}
					allData.add(nv);
					iterations++;
				}
				else if(iterations%2 != 0) {
					for (int i = 0; i < stringLine.length; i++) {
						line1[i] = Integer.valueOf(stringLine[i]);
					}
					allLabel.add(line1[0]);
					iterations++;
				}	
			}
			b.close();
		}
		catch(IOException e) {
			System.out.println(e.getMessage());
		}
	}

	public void initialize() {
		//works
		w1 = null;
		w1 = new double[numOfFeature];
		Random rng = new Random();
		for (int i = 0; i < numOfFeature; i++) {
			// to get a double between -1 and 1
			// rng.nextDouble() is between 0 and 1
			w1[i] = rng.nextDouble() * 2 - 1;
		}
		b1 = rng.nextDouble() * 2 - 1;
	}
	
	public double training() {
		double new_obj = 0, old_obj = 0;
		int MaxIter = 100000;
		int iter = 0;
		int m = trainData.size();
		while(iter < MaxIter) {
			new_obj = 0;
			double[] h = new double[m];
			double[] z = new double[m];
			// Step 1: Compute the h
			double[] pen = new double[2];
			pen[0] = 1;
			pen[1] = 10;
			for(int i = 0; i < m; i++) {
				z[i] = 0;
				for(int nj = 0; nj < trainData.elementAt(i).size(); nj++) {
					int j = trainData.elementAt(i).elementAt(nj);
					z[i] += w1[j];
				}
				z[i] += b1;
				h[i] = 1 / (1 + Math.exp(-z[i]));
			}
			// Step 2: Evaluate the error, given w1
			double error = 0;
			for(int i = 0; i < m; i++) {
				int L = trainLabel.elementAt(i);
				error += pen[L]*(h[i] - trainLabel.elementAt(i))*(h[i] - trainLabel.elementAt(i));
			}
			double reg = 0;
			for(int i = 0; i < numOfFeature; i++) {
				reg += w1[i]*w1[i];
			}
			reg += b1*b1;
			new_obj = (error + lambda*reg)/(2*m);
			System.out.println("Iteration " + iter + " -- Error: " + new_obj);
			if(Math.abs(new_obj - old_obj) < 0.001) break;
			old_obj = new_obj;
			// Step 3: Gradient descent
			double[] dw = new double[numOfFeature];
			double db = 0;
			for(int j = 0; j < numOfFeature; j++) dw[j] = 0;
			for(int i = 0; i < m; i++) {
				int L = trainLabel.elementAt(i);
				for(int nj = 0; nj < trainData.elementAt(i).size(); nj++) {
					int j = trainData.elementAt(i).elementAt(nj);
					dw[j] += pen[L]*(h[i]-trainLabel.elementAt(i)) * (1 / (1 + Math.exp(-z[i]))) * (1 - (1 / (1 + Math.exp(-z[i]))));
				}
				db += pen[L]*(h[i]-trainLabel.elementAt(i)) * (1 / (1 + Math.exp(-z[i]))) * (1 - (1 / (1 + Math.exp(-z[i]))));
			}
			for(int j = 0; j < numOfFeature; j++) {
				dw[j] = (dw[j] + lambda * w1[j])/m;
			}
			db = (db + lambda * b1) / m;
			for(int j = 0; j < numOfFeature; j++) {
				w1[j] = w1[j] - alpha * dw[j];
			}
			b1 = b1 - alpha * db;
			iter++;
		}
		try {
	        BufferedWriter out = new BufferedWriter(new FileWriter("/Users/rdivate2016/Desktop/LogisticParameters-8.txt"));
			for (int i = 0; i < w1.length; i++) {
				out.write(String.valueOf(w1[i]) + "\n");
			}
			out.write(String.valueOf(b1));
	        out.close();
	    } catch (IOException e) {}
		return new_obj;
	}
	
	public void predict() {
		//read parameters from text file
		double[] w = new double[numOfFeature];
		double b2 = 0;
		double threshold = 0.1;
		double numCorrect = 0;
		double totalNum = 0;
		int prediction = 0;
		double numRecall = 0;
		double numToxic = 0;
		double numNonToxic = 0;
		try {
			String line;
			FileReader file = new FileReader("/Users/rdivate2016/Desktop/LogisticParameters-8.txt");
			BufferedReader b = new BufferedReader(file);
			int count = 0;
			while((line = b.readLine()) != null) {
				double value = Double.valueOf(line.split("\n")[0]);
				//System.out.println(value);
				if (count == numOfFeature) {
					b2 = value;
				}
				else {
					w[count] = value;
				}
				count++;
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println(e.getMessage());
		}
		int n = testData.size();
		// Compute prediction
		double[] h = new double[n];
		for (int i = 0 ; i < n; i++) {
			double z = 0;
		    for (int nj = 0; nj < testData.elementAt(i).size(); nj++) {
		        int j = testData.elementAt(i).elementAt(nj);
		    	z += w[j];
		    }
		    z += b2;
	        h[i] = 1 / (1 + Math.exp(-z));
		}
	    //    System.out.println(y);
		//predict using hypothesis function
		totalNum = n;
		for(int i = 0; i < n; i++) {
			if (testLabel.elementAt(i) == 1) {
	        	numToxic += 1;
	        }
	        else {
	        	numNonToxic += 1;
	        }
	        if ( h[i] >= threshold) {
	        	prediction = 1;
	       /* 	System.out.println("Hypothesis:" + h);
	        	System.out.println("Prediction:" + prediction);
	        	System.out.println("Actual:" + y);*/
	        }
	        else {
	        	prediction = 0;
	       /* 	System.out.println("Hypothesis:" + h);
	        	System.out.println("Prediction:" + prediction);
	        	System.out.println("Actual:" + y);*/
	        }
	        if (prediction == testLabel.elementAt(i)) {
	        	if (prediction == 1) {
	        		numRecall += 1;
	        	}
				numCorrect += 1;
			}
		}
        
		System.out.println("The number of correct predictions is " + numCorrect);
		System.out.println("The total number of data pts is " + totalNum);
		System.out.println("The precision is " + numCorrect/totalNum);
		System.out.println("The number of correctly classified toxic compounds is " + numRecall);
		System.out.println("The number of toxic compounds is " + numToxic);
		System.out.println("The recall is " + numRecall/numToxic);
		System.out.println("The number of nontoxic compounds is " + numNonToxic);
		System.out.println("The f score is " +(2.0*(numCorrect/totalNum)*(numRecall/numToxic))/((numCorrect/totalNum)+(numRecall/numToxic)));
	}

	public static void main(String[] args) {
		//System.out.println(java.lang.Runtime.getRuntime().maxMemory());
		String filename = "/Users/rdivate2016/mcf-7-data/mcf-7-gvec-8.txt";
		LogisticClassifier test = new LogisticClassifier();
		test.loadData(filename);
		test.shuffle();
		for(int k = 0; k < 10; k++) {
			// get k-th fold training and test data
			test.makeTrainingData(k);
			//double[][] trainData = test.getTrainData();
			//double[][] testData = test.getTestData();
			
			//test.initialize();
			//test.training();
			test.predict();
		}
	}


}
