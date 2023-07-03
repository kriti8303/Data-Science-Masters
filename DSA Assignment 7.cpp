//******* Assignment Question 1:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool isIsomorphic(string s, string t) {
        int len = s.size();
        char seen[128] = {};
        for(int i=0; i<len; i++){
            char c = s[i];
            if(! seen[c]){
                for(char s:seen){
                    if(s==t[i]){
                        return false;
                    }
                }
                seen[c] = t[i];
            }
            else if(seen[c] != t[i]){
                return false;
            }
        }
        return true;
    }
};

int main(){
    Solution s ;
    vector<string> s, t;
     s = "egg", t = "add";
     vector<string> result= s.isIsomorphic (s, t);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 2:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool isStrobogrammatic(string num) {
        unordered_map<char, char> lut{{'0', '0'}, {'1', '1'}, {'6', '9'}, {'8', '8'}, {'9', '6'}};
        for (int l = 0, r = num.size() - 1; l <= r; l++, r--) {
            if (lut.find(num[l]) == lut.end() || lut[num[l]] != num[r]) {
                return false;
            }
        }
        return true;
    }
};

int main(){
    Solution s ;
    vector<string> v;
     v = "69";
     vector<int> result= s.isStrobogrammatic (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 3:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    string addStrings(string num1, string num2) {
        string res="";
        int c=0,i=num1.size()-1,j=num2.size()-1;
        while(i>=0||j>=0 ||c==1){
            c+=i>=0?num1[i--]-'0':0;
            c+=j>=0?num2[j--]-'0':0;
            res=char(c%10+'0')+res;
            c/=10;
        }
        
        return res;
    }
};

int main(){
    Solution s ;
    vector<string> num1, num2;
     num1 = "11", num2 = "123";
     vector<int> result= s.addStrings (num1, num2);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 4:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    string reverseWords(string s) {
        s.push_back(' ');
        string ans, st;
        for(auto c: s) {
            if(c == ' ') {
                reverse(st.begin(), st.end());
                ans += st;
                ans += ' ';
                st.clear();
            }
            else st.push_back(c);
        }
        ans.pop_back();
        return ans;
    }
};

int main(){
    Solution s ;
    vector<istring> v;
     v = "Let's take LeetCode contest";
     vector<string> result= s.reverseWords (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 5:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    string reverseStr(string s, int k) {
        int n=s.length();
        if(k>n)
        { 
            reverse(s.begin(),s.end());
            return s;
        }
      
       int j=0;
    string ans;
    
    for (int i = 0; i < n; i += (2 * k))
    {
        string temp = s.substr(i, k);
        reverse(temp.begin(), temp.end());
        ans += temp;

        j += k ;
        while (j < n && j < (i+2*k))
        {
            ans.push_back(s[j]);
            j++;
        }
    }
    return ans;
    }
};

int main(){
    Solution s ;
    vector<string> v;
     v="abcdefg";
     vector<int> result= s.reverseStr (v, 2);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 6:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool rotateString(string s, string goal) {
        if(s.size()!=goal.size())return false;
        string concatenate = s+s;
        if(concatenate.find(goal)!=string::npos){
            return true;
        }else return false;
    }
};

int main(){
    Solution s ;
    vector<string> s, goal;
      s = "abcde", goal = "cdeab";
     vector<string> result= s.rotateString(s, goal);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 7:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool backspaceCompare(string s, string t) {
        string sc,tc;
        for(int i=0;i<s.size();i++){
            if(s[i]>='a' && s[i]<='z'){
                sc.push_back(s[i]);
            }
            else{
                if(sc.empty()) continue;
                sc.pop_back();
            }
        }
        for(int i=0;i<t.size();i++){
            if(t[i]>='a' && t[i]<='z'){
                 tc.push_back(t[i]);
            }
            else{
                if(tc.empty()) continue;
                 tc.pop_back();
            }
        }
        if(sc == tc) return 1;
        else return 0;
    }
};

int main(){
    Solution s ;
    vector<string> s, t;
     s = "ab#c", t = "ad#c";
     vector<string> result= s.backspaceCompare (s, t);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 8:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    bool checkStraightLine(vector<vector<int>>& coordinates) {
        int x0 = coordinates[0][0];
        int y0 = coordinates[0][1];
        int x1 = coordinates[1][0];
        int y1 = coordinates[1][1];
        
        for (int i = 2; i < coordinates.size(); i++) {
            int x = coordinates[i][0];
            int y = coordinates[i][1];
            if ((x - x0) * (y1 - y0) != (y - y0) * (x1 - x0)) {
                return false;
            }
        }
        
        return true;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v=[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]];
     vector<int> result= s.checkStraightLine (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}
