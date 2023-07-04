//******* Assignment Question 1:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int minimumDeleteSum(string s1, string s2) {
        int n = s1.size();
        int m = s2.size();
        int sum1 = 0, sum2 = 0;
        for(auto i:s1){
            sum1 += i;
        }
        for(auto i:s2){
            sum2 += i;
        }
        vector<int> prev(m+1, 0), curr(m+1, 0);
        for(int i=1; i<=n; i++){
            for(int j=1; j<=m; j++){
                if(s1[i-1] == s2[j-1]){
                    curr[j] = s1[i-1] + prev[j-1];
                }
                else{
                    curr[j] = max(prev[j], curr[j-1]);
                }
            }
            prev = curr;
        }
        return (sum1+sum2) - 2*prev[m];
    }
};

int main(){
    Solution s ;
    vector<string> s1, s2;
     s1 = "sea", s2 = "eat";
     vector<string> result= s.minimumDeleteSum (s1, s2);
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
    bool checkValidString(string s) {
        int leftMin = 0;
        int leftMax = 0;
        
        for (int i = 0; i < s.size(); i++) {
            if (s[i] == '(') {
                leftMin++;
                leftMax++;
            } else if (s[i] == ')') {
                leftMin--;
                leftMax--;
            } else {
                leftMin--;
                leftMax++;
            }
            if (leftMax < 0) {
                return false;
            }
            if (leftMin < 0) {
                leftMin = 0;
            }
        }
        if (leftMin == 0) {
            return true;
        }
        return false;
    }
};

int main(){
    Solution s ;
    vector<string> v;
    v = "()";
     vector<string> result= s.checkValidString (v);
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
    int minDistance(string word1, string word2) {
        int n=word1.length(),m=word2.length();
        int dp[n+1][m+1];
        for(int i=0;i<=n;i++){
            for(int j=0;j<=m;j++){
                if(i==0 || j==0){dp[i][j]=0;}
                else{
                    if(word1[i-1]==word2[j-1]){
                        dp[i][j]=1+dp[i-1][j-1];
                    }
                    else{
                        dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
                    }
                }
            }
        }
        return n+m-2*dp[n][m];
    }
};

int main(){
    Solution s ;
    vector<string> word1, word2;
     word1 = "sea", word2 = "eat";
     vector<string> result= s.minDistance(word1, word2);
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
    string tree2str(TreeNode* root) {
        string result = "";
        if (!root)
            return result;

        result += std::to_string(root->val);
        if (root->left && !root->right) {
            result += "(" + tree2str(root->left) + ")";
        } else if (root->right){
            result += "(" + tree2str(root->left) + ")";
            result += "(" + tree2str(root->right) + ")";
        }
           
        return result;
    }
};

int main(){
    Solution s ;
    vector<string> v;
     v = "4(2(3)(1))(6(5))";
     vector<string> result= s.tree2str (v);
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
    int compress(vector<char>& chars) {
        string s= "";
        for(auto x:chars)
        {
            s+=x;
        }
      
        if(s.size()==1)
        {
            return 1;
        }
       
        int count=1, ans=0;;
        char ch;
        vector<char>a;
        int n=s.size();
        for(int i=0;i<n-1;i++)
        {
            if(s[i]!=s[i+1])
            { 
               a.push_back(s[i]);
                 if(count>1)
               {
                   ans+=2;
                    string temp=to_string(count);
                  
                    for(int it=0;it<temp.size();it++)
                      a.push_back(temp[it]);
                }
               else
                ans+=1;
               count=1;

                 
            }
            else
            { 
                ch=s[i];
                count++;

            }

        }
        a.push_back(s[n-1]);
        if(s[n-1]==s[n-2])
        {
            
                string temp=to_string(count);
                    for(int it=0;it<temp.size();it++)
                      a.push_back(temp[it]);
          
        }
       
        chars=a;
        
        return a.size();
    }
};

int main(){
    Solution s ;
    vector<string> chars;
     chars = ["a","a","b","b","c","c","c"];
     vector<string> result= s.compress (chars);
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
    vector<int> findAnagrams(string s, string p) {
        vector<int> freq_p(26), freq_s(26);
        for(auto &it:p) freq_p[it-'a']++;

        int low=0;
        vector<int> res;
        for(int high=0; high<s.size(); high++)
        {
            freq_s[s[high]-'a']++;
            if((high-low+1==p.size()) && (freq_p==freq_s))
                res.push_back(low);

            if(high-low+1 >= p.size())
            {
                freq_s[s[low]-'a']--;
                low++;
            }
        }
        return res;
    }
};

int main(){
    Solution s ;
    vector<string> s, p;
     s = "cbaebabacd", p = "abc";
     vector<string> result= s.findAnagrams (s, p);
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
    string decodeString(string s) {
        stack<string> st;
        for(auto ch: s){
            if(ch==']'){
                string stringtorepeat="";
                while(!st.empty()&& st.top()!="["){
                    string top=st.top();
                    stringtorepeat+=top;
                    st.pop();
                }
                st.pop();//st.top has '[' this one the top so pop
                //now we will take number of time we want to repeat 
                string num="";
                while(!st.empty() && isdigit(st.top()[0])){
                    string top=st.top();
                    num+=st.top();
                    st.pop();
                }
                // reverse 321 to 123
                reverse(num.begin(),num.end());
                int n=stoi(num);
            

                //final decoding 
                string currdecode="";
                while(n--){
                    currdecode+=stringtorepeat;
                }
                st.push(currdecode);
            }
            else 
            {
                string temp(1,ch);///convert char to string
                st.push(temp);
            }
        }
        ///ye mujhe sari stirng dedega abh m inko final rev karke ans return kardunga 
                string ans="";
                while(!st.empty()){
                    ans+=st.top();
                    st.pop();
                }
                reverse(ans.begin(),ans.end());
                return ans;
    }
};

int main(){
    Solution s ;
    vector<string> s;
     s = "3[a]2[bc]";
     vector<string> result= s.decodeString (s);
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
    bool buddyStrings(string s, string goal) {
        int n = s.length();
        if(s == goal){
            set<char> temp(s.begin(), s.end());
            return temp.size() < goal.size(); // Swapping same characters
        }

        int i = 0;
        int j = n - 1;

        while(i < j && s[i] == goal[i]){
            i++;
        }

        while(j >= 0 && s[j] == goal[j]){
            j--;
        }

        if(i < j){
            swap(s[i], s[j]);
        }

        return s == goal;
    }
};

int main(){
    Solution s ;
    vector<string> s, goal;
     s = "ab", goal = "ba";
     vector<string> result= s.buddyStrings (s, goal);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


