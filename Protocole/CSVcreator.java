package beingSeing;

import java.io.*;

public class CSVcreator {
	final protected String SEPARATEUR = "\n" ;
	final protected String DELIMITEUR = "," ;
	final protected String ENTETE = "Boutton, Temps(sec)";

	protected String csvPath;
	protected File csvFile ; 
	protected boolean enterFile ; 

	public CSVcreator(String SavePath, String nomParticipant, String nomAgent) {
		// constructeur : ouverture du fichier au format de chemin Windows
		csvPath = SavePath + "\\"+ nomParticipant+nomAgent + ".csv";
		csvFile = new File(csvPath);
		enterFile = true ;
	}

	public void sendToCSV(String nomBoutton, String temps) throws IOException {
		// Méthode : enregistrement des valeurs sous forme de CSV
		
		try (BufferedWriter out =  new BufferedWriter(new FileWriter(csvFile, true))) {
			if (enterFile) {
				out.write(ENTETE+SEPARATEUR);
				enterFile = false ; 
			}
			String toWrite = nomBoutton+DELIMITEUR+temps+SEPARATEUR; 
			out.write(toWrite) ;
			System.out.println(toWrite) ;
			out.close();
		} catch (IOException e) {
			e.printStackTrace() ; 
			System.err.print("Valeur non enregistrée dans le CSV : "+ nomBoutton+" à " + temps);
		}
		
	}
	


	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		CSVcreator newCSV = new CSVcreator("C:\\Users\\MickaellaGV.DESKTOP-GDFH7VA\\Documents","MickaTest", "Agent1") ; 
		newCSV.sendToCSV("Boutton2", "3");
		newCSV.sendToCSV("Boutton3", "3");
		newCSV.sendToCSV("Boutton4", "3");
		newCSV.sendToCSV("Boutton5", "3");
	}

}
