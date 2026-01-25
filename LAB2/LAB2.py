import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def purchasecategorize():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")

    df["Label"] = np.where(df["Payment (Rs)"] > 200, 1, 0)

    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Label"].values

    X = np.column_stack((np.ones(len(X)), X))

    W = np.linalg.pinv(X) @ y
    pred = X @ W
    prediction = np.where(pred >= 0.5, 1, 0)

    df["Prediction"] = np.where(prediction == 1, "RICH", "POOR")

    print(df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)", "Payment (Rs)", "Prediction"]])

def purchase():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    x = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = df['Payment (Rs)'].values
    print("Matrix X : " , x.shape)
    print("Matrix Y : ",y.shape)
    rankx = np.linalg.matrix_rank(x)
    print("Rank of feature matrix X:", rankx)
    xpinv = np.linalg.pinv(x)
    c = xpinv @ y
    print("Cost of Candies : ", c[0])
    print("Cost of Mangoes : ", c[1])
    print("Cost of Milk Packets : ", c[2])

def meanx():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    Price = df[['Price']].values
    m = 0
    for i in range(len(Price)):
        m += Price[i][0]
    meanx = m/len(Price)
    return meanx

def variancex():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    Price = df[['Price']].values
    mx = meanx()
    v = 0
    for i in range(len(Price)):
        v += (Price[i][0] - mx)**2
    variance = v/len(Price)-1
    return variance

def IRTC():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
    Price = df[['Price']].values
    chg = df[['Chg%']].values
    
    print(f"Mean of price (using numpy): {np.mean(Price)}")
    print(f"Mean of price (using numpy): {np.var(Price)}")
    print(f"Mean of price (without numpy): {meanx()}")
    print(f"Variance of price (without numpy): {variancex()}")

    meanwed = df[df["Day"]=="Wed"]["Price"].mean()
    print(f"Sample mean of price data for all wednesdays : {meanwed}")
    
    df["Date"] = pd.to_datetime(df["Date"])
    meanapril = df[df["Date"].dt.month==4]["Price"].mean()
    print(f"Sample mean of price data for the month April : {meanapril}")
    
    loss = len(list(filter(lambda x: x < 0, chg))) / len(chg)
    print(f"Probability of making a loss over the stock : {loss}")
    
    wedchg = df[df["Day"]=="Wed"]["Chg%"]
    print(f"probability of making a profit on Wednesday : {wedchg}")
    
    profitwed = len(wedchg[wedchg > 0]) / len(wedchg)
    print(f"probability of making profit, given that today is Wednesday : {profitwed}")
    
    plt.figure()
    plt.scatter( df["Day"],df["Chg%"])
    plt.xlabel("Day of Week")
    plt.ylabel("Chg%")
    plt.title("Scatter plot of Chg% vs Day of Week")
    plt.show()

def thyroid():

    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    df = df.replace("?", np.nan)

    numcols = df.select_dtypes(include=np.number).columns.drop("Record ID")
    catcols = df.select_dtypes(include="object").columns

    for col in numcols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        if outliers > 0:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    for col in catcols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df_norm = df.copy()

    for col in numcols:
        df_norm[col] = (df[col] - df[col].mean()) / df[col].std()

    print("Normalized Data:")
    print(df_norm[numcols].head())



def SMC():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    v1 = df.iloc[0]
    v2 = df.iloc[1]

    bincols = []
    for col in df.columns:
        if df[col].dtype == object:
            vals = set(df[col].dropna().unique())
            if vals.issubset({"f","t"}):
                bincols.append(col)

    v1bin = v1[bincols].map({"f":0,"t":1})
    v2bin = v2[bincols].map({"f":0,"t":1})

    f11 = np.sum((v1bin==1)&(v2bin==1))
    f10 = np.sum((v1bin==1)&(v2bin==0))
    f01 = np.sum((v1bin==0)&(v2bin==1))
    f00 = np.sum((v1bin==0)&(v2bin==0))

    SMC = (f11+f00)/(f00+f01+f10+f11)

    print(f"f00 : {f00}")
    print(f"f01: {f01}")
    print(f"f10 : {f10}")
    print(f"f11 : {f11}")
    return SMC

def JC():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    v1 = df.iloc[0]
    v2 = df.iloc[1]

    bincols = []
    for col in df.columns:
        if df[col].dtype == object:
            vals = set(df[col].dropna().unique())
            if vals.issubset({"f","t"}):
                bincols.append(col)

    v1bin = v1[bincols].map({"f":0,"t":1})
    v2bin = v2[bincols].map({"f":0,"t":1})

    f11 = np.sum((v1bin==1)&(v2bin==1))
    f10 = np.sum((v1bin==1)&(v2bin==0))
    f01 = np.sum((v1bin==0)&(v2bin==1))
    f00 = np.sum((v1bin==0)&(v2bin==0))

    JC = f11/(f01+f10+f11)

    print(f"f00 : {f00}")
    print(f"f01: {f01}")
    print(f"f10 : {f10}")
    print(f"f11 : {f11}")

    return JC

def cosine():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    v = df.iloc[0:2].copy()
    v = v.replace("?", np.nan)

    for col in v.columns:
        vals = set(v[col].dropna().unique())
        if vals.issubset({"f","t"}):
            v[col] = v[col].map({"f":0,"t":1})
    for col in v.columns:
        if v[col].dtype == object:
            v[col], _ = pd.factorize(v[col])

    v = v.fillna(0)
    A = v.iloc[0].values.astype(float)
    B = v.iloc[1].values.astype(float)
    dotp = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine = dotp / (norm_A * norm_B)

    return cosine

def heatmap():

    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")

    df20 = df.iloc[:20].copy()
    n = len(df20)

    bincols = []
    for col in df20.columns:
        if df20[col].dtype == object:
            vals = set(df20[col].dropna().unique())
            if vals.issubset({"f","t"}):
                bincols.append(col)
    bindata = df20[bincols].replace({"f":0,"t":1})

    JC = np.zeros((n,n))
    SMC = np.zeros((n,n))
    cos = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            v1 = bindata.iloc[i]
            v2 = bindata.iloc[j]

            f11 = np.sum((v1==1)&(v2==1))
            f10 = np.sum((v1==1)&(v2==0))
            f01 = np.sum((v1==0)&(v2==1))
            f00 = np.sum((v1==0)&(v2==0))

            JC[i,j] = f11/(f01+f10+f11) if (f01+f10+f11)!=0 else 0
            SMC[i,j] = (f11+f00)/(f00+f01+f10+f11)

    dfcos = df20.replace("?", np.nan)

    for col in dfcos.columns:
        vals = set(dfcos[col].dropna().unique())
        if vals.issubset({"f","t"}):
            dfcos[col] = dfcos[col].map({"f":0,"t":1})

    for col in dfcos.columns:
        if dfcos[col].dtype == object:
            dfcos[col], _ = pd.factorize(dfcos[col])

    dfcos = dfcos.fillna(0)
    X = dfcos.values.astype(float)

    for i in range(n):
        for j in range(n):
            dot = np.dot(X[i], X[j])
            normi = np.linalg.norm(X[i])
            normj = np.linalg.norm(X[j])
            cos[i,j] = dot/(normi*normj)

    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    sns.heatmap(JC, cmap="Blues")
    plt.title("Jaccard Coefficient Heatmap")

    plt.subplot(1,3,2)
    sns.heatmap(SMC, cmap="Greens")
    plt.title("Simple Matching Coefficient Heatmap")

    plt.subplot(1,3,3)
    sns.heatmap(cos, cmap="Reds")
    plt.title("Cosine Similarity Heatmap")

    plt.tight_layout()
    plt.show()

def main():

    while(True):
        print("1. Purchase Data ")
        print("2. Purchase Data categorization")
        print("3. IRTC Stock Price ")
        print("4. Data exploration , Data imputation , Normalization ")
        print("5. Simple matching coefficient (SMC)")
        print("6. Jaccard coeffecient (JC)")
        print("7. Cosine similarity ")
        print("8. Heatmap")
        ch = int(input("Enter your choice : "))
        if ch == 1:
            purchase()
        elif ch == 2:
            purchasecategorize()
        elif ch == 3:
            IRTC()
        elif ch == 4:
            thyroid()
        elif ch == 5:
            print(f"Simple matching coefficient (SMC) : {SMC()}")
        elif ch == 6:
            print(f"Jaccard coefficient (JC) : {JC()}")
        elif ch == 7:
            print(f"Cosine similarity : {cosine()}")
        elif ch == 8:
            heatmap()
        else:
            print("Invalid choice")

main()


