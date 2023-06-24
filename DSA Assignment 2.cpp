//******* Assignment Question 1 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        int ans = 0;
        sort(nums.begin(), nums.end());
        for(int i=0; i<nums.size(); i+=2){
            ans += nums[i];
        }
        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,4,3,2};
     vector<int> result= s.arrayPairSum (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 2 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int distributeCandies(vector<int>& candyType) {
        unordered_set<int> unique_candies;
        for(int i=0; i<candyType.size(); i++){
            unique_candies.insert(candyType[i]);
        }
        if((unique_candies.size())<=(candyType.size()/2)){
            return unique_candies.size();
        }
        return candyType.size()/2;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,1,2,2,3,3};
     vector<int> result= s.distributeCandies (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

//******* Assignment Question 3 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int findLHS(vector<int>& nums) {
       unordered_map<int, int> m;
       int maxsum = 0;
       for(auto num : nums){
           ++m[num];
       }
       for(auto& [num, value] : m){
           if(m.end() != m.find(num-1)){
               if(maxsum < m[num-1]+value){
                   maxsum = m[num-1] + value;
               }
           }
       }
       return maxsum;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,3,2,2,5,2,3,7};
     vector<int> result= s.findLHS (v);
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
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        for(int i=0; i<flowerbed.size(); i++){
            if(flowerbed[i]==0){
                if((i==0 || flowerbed[i-1]==0) && (i==flowerbed.size()-1 || flowerbed[i+1]==0)){
                    n--;
                    flowerbed[i]=1;
                }
            }
        }
        return (n<=0);
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,0,0,0,1};
     vector<int> result= s.canPlaceFlowers (v, 1);
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
    int maximumProduct(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        return max(nums[0]*nums[1]*nums[n-1], nums[n-1]*nums[n-2]*nums[n-3]);
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3};
     vector<int> result= s.maximumProduct (v);
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
    int search(vector<int>& nums, int target) {
        int s = 0;
        int e = nums.size()-1;
        while(s<=e){
            int mid = s+(e-s)/2;
            if(nums[mid]==target){
                return mid;
            }
            else if(target<nums[mid]){
                e = mid-1;
            }
            else{
                s = mid+1;
            }
        }
        return -1;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={-1,0,3,5,9,12};
     vector<int> result= s.search (v, 9);
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
    bool isMonotonic(vector<int>& nums) {
        bool increase = true;
        bool decrease = true;
        for(int i=0; i<nums.size()-1; i++){
            if(nums[i]>nums[i+1]){
                increase = false;
            }
            if(nums[i]<nums[i+1]){
                decrease = false;
            }
            if(increase == false && decrease == false){
                return false;
            }
        }
        return true;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,2,3};
     vector<int> result= s.isMonotonic (v);
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
    int smallestRangeI(vector<int>& nums, int k) {
        int mx = nums[0];
        int mn = nums[0];
        for(int a: nums){
            mx = max(mx, a);
            mn = min(mn, a);
        }
        return max(0, mx-mn-2*k);
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1};
     vector<int> result= s.smallestRangeI (v, 0);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}