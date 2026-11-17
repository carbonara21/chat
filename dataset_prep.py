
import pandas as pd

# 1. Load dataset
df = pd.read_csv(
    "/Users/felipecarbone/PycharmProjects/CHAT/pokemon_chat_annotations_anonymized.csv"
)

# 2. Remove unnecessary columns
df = df.drop(columns=['time', 'channel'])

# 3. Identify annotation columns
X = df.columns.get_loc('Battle_Win')
Y = df.columns.get_loc('Collective_Theory_Building')

category_cols = df.columns[X:Y+1]

# 4. Transform wide format into long format
df = df.melt(
    id_vars=['message'],
    value_vars=category_cols,
    var_name='category',
    value_name='value'
)

# 5. Keep only active annotations (value = 1)
df = df[df['value'] == 1]

# 6. Remove unnecessary value column
df = df.drop(columns=['value'])

# 7. Reset index
df = df.reset_index(drop=True)

# 8. Save dataset
df.to_csv("pokemon_prepared.csv", index=False)

print(df.head(20))
print(df['category'].value_counts())
