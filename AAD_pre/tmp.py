
def a():
    import statsmodels.api as sm
    import pandas as pd
    from statsmodels.formula.api import ols
    num = sorted(['g1', 'g2', 'g3','g4', 'g5']*4)  
    data = group1 + group2 + group3 + group4 + group5  
    df = pd.DataFrame({'num':num, 'data': data})  
    mod = ols('data ~ num', data=df).fit()          
    ano_table = sm.stats.anova_lm(mod, typ=2)  
    print(ano_table) 
def a2():
    from scipy import stats  
    F, p = stats.f_oneway(group1, group2, group3, group4, group5)  
    F_test = stats.f.ppf((1-0.05), 4, 15) 
    print('F值是%.2f，p值是%.9f' % (F,p))  
    print('F_test的值是%.2f' % (F_test))  
    if F>=F_test:  
        print('拒绝原假设，u1、u2、u3、u4、u5不全相等')  
    else:  
        print('接受原假设，u1=u2=u3=u4=u5') 
