// Standard C++ libraries
#include<iostream>
#include<math.h>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>


using namespace std;

// Loading the data (WMT-14)
vector<vector<string> > readCSV(const string& filename){

  ifstream file(filename);
  vector<vector<string> > data;
  string line;

  if(!file.is_open()){
    cout<<"Couldn't load data";
    return data;
  }
  
  while(getline(file, line, '\n')){
      stringstream ss(line);
      
      vector<string> tokens;
      string token;
      while(getline(ss, token, ',')){
        tokens.push_back(token);
      }
      data.push_back(tokens);
  } 
  file.close();

  return data;
}

//
// Model Architecture



int main() { 
  
  // loading the data
  //string file = "../datasets/wmt-14/wmt14_translate_de-en_train.csv";
  string file = "../datasets/test.csv";
  vector<vector<string> > data = readCSV(file);

  for(int i=0;i<data.size();i++){
    for(string val:data[i]){
      cout<<val;
    }
    cout<<endl;
  }


  return 0; 

}
