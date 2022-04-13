from scipy.stats import multivariate_normal

male = multivariate_normal(mean=[172.50826135, 68.89546617], cov=[[33.77298128, 27.24172068], [27.24172068, 44.17279407]])
female = multivariate_normal(mean=[157.56548559, 55.7903368], cov=[[48.02290099, -7.5190101], [-7.5190101, 41.4043647]])

# 给定数据集[身高，体重]
data = [165, 62]

# 先验概率
p_male = 0.5
p_female = 1 - p_male

# 似然度模型
x = male.pdf(data)
y = female.pdf(data)

# 后验概率
pos_male = x*p_male / (x*p_male + y*p_female)
pos_female = y*p_female / (x*p_male + y*p_female)

print("Likelihood of male: " + str(x * 100))
print("Likelihood of female: " + str(y * 100))

print("Posterior probability p(男|数据) of male: " + str(x*p_male / (x*p_male + y*p_female)))
print("Posterior probability p(女|数据) of female: " + str(y*p_female / (x*p_male + y*p_female)))
