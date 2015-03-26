package weka.classifiers.bayes;


import java.util.LinkedList;

import weka.core.Instances;

public class Node {

	private Node father;
	private Instances node;
	private Node leftChildren;
	private Node rightChildren;
	private boolean checked;
	private String position;
	private NaiveBayesMultinomial nb;
	private int sizeH;
	Node (Instances nNode,Node nFather, String pos) throws Exception{
		leftChildren=null;
		rightChildren=null;
		setChecked(false);
		father=nFather;
		position=pos;
		node=nNode;
		//System.out.println(node.numInstances());
		nb = new NaiveBayesMultinomial();
		nb.buildClassifier(node);
		sizeH = node.numInstances();

	}
	
	public Node getLeftChildren(){
		return leftChildren;
	}
	
	public Node getRightChildren(){
		return rightChildren;
	}
	
	public double getNumNodes(){
		int h=0; //current Node
		LinkedList<Node> nodes = new LinkedList<Node>();
		nodes.add(this);
		while (nodes.size()!=0){
			Node nd=nodes.removeFirst();
			h++;
			if (nd.leftChildren!=null)
				nodes.addLast(nd.leftChildren);
			if (nd.rightChildren!=null)
				nodes.addLast(nd.rightChildren);
		}
		return h;
	}
	
	public Instances getInstances(){
		return node;
	}
	
	public double getSizeH(){
		int h=0; //current Node
		double sizeH;
		LinkedList<Node> nodes = new LinkedList<Node>();
		nodes.add(this);
		while (nodes.size()!=0){
			Node nd=nodes.removeFirst();
			h++;
			if (nd.leftChildren!=null)
				nodes.addLast(nd.leftChildren);
			if (nd.rightChildren!=null)
				nodes.addLast(nd.rightChildren);
		}
		sizeH = (node.numClasses() + (node.numClasses()*node.numAttributes()))*h;
		return sizeH;
	}
	
	public double getCCL() throws Exception{
		double cll = 0;
			for (int i = 0; i < node.numInstances(); i++) {
				if (node.instance(i).classValue() == 0) {
					cll += Math.log(nb.distributionForInstance(node.instance(i))[0]);
				} else {
					cll += Math.log(nb.distributionForInstance(node.instance(i))[1]);
				}
			}
		cll*=node.numInstances();
		if (!hasChild())
			return cll;
		else{
			double finalCll=0;
			if (leftChildren!=null)
				finalCll+=leftChildren.getCCL();
			if (rightChildren!=null)
				finalCll+=rightChildren.getCCL();
			return finalCll;
		}
	}

	
	public double getCMDL() throws Exception{
		double sizeH = getSizeH();
		double cmdl = getCCL() - (Math.log(node.numInstances())/2) * sizeH;
		return cmdl;
	}

	public boolean hasChild(){
		if (leftChildren==null && rightChildren==null )
			return false;
		return true;
	}
	
	public boolean split() throws Exception{
		Instances left = new Instances(node, 0);
		Instances right = new Instances(node, 0);
		for (int i=0;i<node.numInstances();i++){
			double instClass= nb.classifyInstance(node.instance(i));
			if (instClass<0.5)
				left.add(node.instance(i));
			else
				right.add(node.instance(i));
		}
		if (left.numInstances()>0 && right.numInstances()>0){
			Node leftChild= new Node(left,this,"left");
			addLeftChildren(leftChild);
			Node rightChild= new Node(right,this,"right");
			addRightChildren(rightChild);
			return true;
		}
		else 
			return false;
	}
	
	public void revertBack(){
		leftChildren=null;
		rightChildren=null;
		setChecked(true);
	}
	
	public void addRightChildren(Node child){
		rightChildren=child;
	}
	
	public void addLeftChildren(Node child){
		leftChildren=child;
	}
		
	public Node getFather() {
		return father;
	}
	
	public void setFather(Node father) {
		this.father = father;
	}
	
	public void deleteLeftChildren() {
		leftChildren=null;
	}
	
	public void deleteRightChildren() {
		rightChildren=null;
	}
	
	public double getClassAfterClassified(){
		if (position=="right")
			return 1;
		else
			return 0;
	}
	
	public boolean getChecked() {
		return checked;
	}

	public String getPosition() {
		return position;
	}
	
	public void setChecked(boolean checked) {
		this.checked = checked;
	}

}

