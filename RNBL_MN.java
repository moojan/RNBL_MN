package weka.classifiers.bayes;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.DataSource;


public class RNBL_MN extends Classifier{

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private Node root;
	private LinkedList<Node> leaves = new LinkedList<Node>();
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		root=new Node(data,null,"root");
		LinkedList<Node> nodes = new LinkedList<Node>();
		nodes.add(root);
		double currentCMDL=root.getCMDL();
		double lastCMDL=Double.NEGATIVE_INFINITY;
		
		
		while (!nodes.isEmpty()){
			Node nd=nodes.removeFirst();
			if (!nd.getChecked()){
				boolean splitted = nd.split();
				if (splitted){
					lastCMDL=currentCMDL;
					currentCMDL=nd.getCMDL();
			//	System.out.print(lastCMDL);
			//	System.out.print(currentCMDL);
			//	System.out.print("\n");
					if (currentCMDL>lastCMDL){
						if (nd.getLeftChildren()!=null)
							nodes.addLast(nd.getLeftChildren());
						if (nd.getRightChildren()!=null)
							nodes.addLast(nd.getRightChildren());
					}
					else{
						nd.revertBack();
						leaves.addLast(nd);
					}
				}
				else{
					nd.setChecked(true);
					leaves.addLast(nd);
				}
			}
		}
	//	System.out.println(root.getNumNodes());
	//	System.out.println(leaves.size());
	//	System.out.println(leaves);
		
	}

	public void testClassifier(String fileName) throws FileNotFoundException, UnsupportedEncodingException{
		double tp=0;
		double fp=0;
		double tn=0;
		double fn=0;
		for (Node nd : leaves){
			Instances instances=nd.getInstances();
			for (int i=0;i<instances.numInstances();i++){
				if (instances.instance(i).classValue()==nd.getClassAfterClassified()){
					if(instances.instance(i).classValue()==1)
						tp++;
					else
						tn++;
				}
				else{
					if(nd.getClassAfterClassified()==0)
						fn++;
					else
						fp++;		
				}
			}
		}
		double precision = tp / (tp+fp);
		double recall = tp / (tp+fn);
		double accuracy = (tp+tn) / (tp+fp+tn+fn);
		double fmeasure = (2*precision*recall) / (precision + recall);
		PrintWriter writer = new PrintWriter("Lab2ResultsPartC/"+fileName+"Results.txt", "UTF-8");
		writer.println("Number of nodes: "+root.getNumNodes());
		writer.println("Number of leaves: "+leaves.size());
		writer.println("True Positive:"+tp);
		writer.println("False Positive:"+fp);
		writer.println("True Negative:"+tn);
		writer.println("False Negative:"+fn);
		writer.println("Precision: "+precision);
		writer.println("Recall: "+recall);
		writer.println("accuracy: "+accuracy);
		writer.println("F-measure: "+fmeasure);
		writer.close();

	}
	
	
	public static void main(String [] argv) throws Exception {
		String dataAddress=System.getProperty("user.dir");
		String[] trainFiles={"acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade", "wheat"};
		// read train set
		for (String fileName : trainFiles){
			BufferedReader reader = new BufferedReader( new FileReader(dataAddress+"/finalData/"+fileName+"-train.arff"));
			Instances train = new Instances(reader);
			reader.close();
			// setting class attribute
			train.setClassIndex(train.numAttributes() - 1);
			//train the data
			RNBL_MN rnbl= new RNBL_MN();
			rnbl.buildClassifier(train);
			//test the classifier and write the output to the file
			rnbl.testClassifier(fileName);
		}
		
	}




	

}
