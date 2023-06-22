//******* Assignment Question 1 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for(int i=0; i<n; i++){
            for(int j=0; j<n; j++){
                if(i!=j && nums[i] + nums[j] == target){
                    //print the pair
                    return {i, j};
                }
            }
        }
        return {};
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={2,7,11,15};
     vector<int> result= s.twoSum (v, 9);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

// ****** Assignment Question 2 :

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
	int removeElement(vector<int> &nums, int val){
		vector<int> :: iterartor it;
		it = nums.begin();
		for(int i=0; i<nums.size(); i++){
			if(nums[i]==val){
				nums.erase(it);
				it--;
				i--;
			}
			it++;
		}
		return nums.size();
	}
};

int main(){
	Solution s;
    vector<int> v;
     v={3, 2, 2, 3};
     int result= s.removeElement(v, 3);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


// ********Assignment Question 3:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int s = 0;
        int e = nums.size()-1;
        while(s<=e){
            int mid = s+(e-s)/2;
            if(nums[mid]==target){
                return mid;
            }
            if(target<nums[mid]){
                e = mid-1;
            }
            else{
                s = mid+1;
            }
        }
        return s;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,3,5,6};
     vector<int> result= s.searchInsert (v, 5);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

// ********Assignment Question 4:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> plusOne(vector<int>& digit){
        int n = digit.size();
        for(int i=n-1; i>=0; i--){
            if(digit[i]<9){
                digit[i]++;
                return digit;
            }
            else{
                digit[i] = 0;
            }
        }
        digit.push_back(0);     
        digit[0] = 1;
        return digit;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3};
     vector<int> result= s.plusOne (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}


// ********Assignment Question 5:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int i = m-1, j = n-1;
        int k = m+n-1;
        while(i>=0 && j>=0){
            if(nums1[i]>nums2[j]){
                nums1[k] = nums1[i];
                i--;
                k--;
            }
            else{
                nums1[k]=nums2[j];
                j--;
                k--;
            }
        }
        while(i>=0){
            nums1[k--]=nums1[i--];
        }
        while(j>=0){
            nums1[k--]=nums2[j--];
        }
    }
};

int main(){
    Solution s ;
    vector<int> num1, num2;
     num1={1,2,3,0,0,0};
     num2={2,5,6};
     vector<int> result= s.merge (num1, 3, num2, 3);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

// ********Assignment Question 6:

# include<iostream>
# include<vector>
using namespace std;

class Solution{
public:
	int findDuplicates(vector<int> &arr){
		int ans = 0;
		for(int i=0; i<arr.size(); i++){
			ans = ans^arr[i];
		}
		for(int i=0;  i<arr.size(); i++){
			ans = ans^i;
		}
		return ans;
	}	
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,3,1};
     vector<int> result= s.merge (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

// ********Assignment Question 7:

# include<iostream>
# include<vector>
using namespace std;

class Solution{
public:
	int moveZeros(vector<int> &nums){
		int i=0;
		for(int j=0; j<nums.size(); j++){
			if(nums[j]!=0){
				swap(nums[j], nums[i]);
				i++;
			}
		}
	}
};


int main(){
    Solution s ;
    vector<int> v;
     v={0,1,0,3,12};
     vector<int> result= s.moveZeros (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}

// ********Assignment Question 8:

# include<iostream>
# include<vector>
using namespace std;

class Solution {
public:
    vector<int> findErrorNums(vector<int>& nums) {
        vector<int> ans;
        unordered_set<int> s;
        int sum = 0;
        for(auto x:nums){
            if(s.find(x)!=s.end()){
                ans.push_back(x);
            }
            else{
                s.insert(x);
                sum+=x;
            }
        }
        int n = nums.size();
        int t = n*(n+1)/2;
        ans.push_back(t-sum);

        return ans;
    }
};

int main(){
    Solution s ;
    vector<int> v;
     v={1,2,2,4};
     vector<int> result= s.moveZeros (v);
     for(int i: result)
     {
          cout<<i<<endl;
     }
     return 0;
}