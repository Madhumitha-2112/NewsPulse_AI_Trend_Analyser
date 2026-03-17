import pandas as pd
import re

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------

file_path = r"C:\Users\Welcome\Desktop\Newspulse_Virtual_Internship\multi_genre_news_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!\n")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()



df.columns = df.columns.str.strip().str.lower()

print("Columns in Dataset:")
print(df.columns)
print("\n")

# -------------------------------------------------
# Step 3: Display Sample Headlines
# -------------------------------------------------

print("Sample News Headlines:\n")

if 'title' in df.columns:
    for i, headline in enumerate(df['title'].head(5), 1):
        print(f"{i}. {headline}")
else:
    print("Error: 'title' column not found in dataset.")

print("\n")

# -------------------------------------------------
# Step 4: Total News Articles
# -------------------------------------------------

total_articles = len(df)
print("Total News Articles:", total_articles)

# -------------------------------------------------
# Step 5: Unique News Sources
# -------------------------------------------------

if 'source_name' in df.columns:
    unique_sources = df['source_name'].nunique()
    print("Unique News Sources:", unique_sources)
else:
    print("Error: 'source_name' column not found.")

print("\n")

# -------------------------------------------------
# Step 6: Data Cleaning
# -------------------------------------------------

print("Performing Data Cleaning...\n")

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Clean special characters from titles
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text))

if 'title' in df.columns:
    df['title'] = df['title'].apply(clean_text)

print("Data Cleaning Completed Successfully!\n")

# -------------------------------------------------
# Step 7: Save Cleaned Dataset
# -------------------------------------------------

cleaned_file_path = r"C:\Users\Welcome\Desktop\Newspulse_Virtual_Internship\cleaned_news_dataset.csv"

df.to_csv(cleaned_file_path, index=False)

print("Cleaned CSV File Generated Successfully!")
print(f"Saved at: {cleaned_file_path}\n")

# -------------------------------------------------
# Step 8: Final Summary
# -------------------------------------------------

print("===== NewsPulse Project Summary =====\n")

print("Total News Articles After Cleaning:", len(df))

if 'source_name' in df.columns:
    print("Unique News Sources:", df['source_name'].nunique())

print("\nSample Cleaned Headlines:\n")

if 'title' in df.columns:
    for i, headline in enumerate(df['title'].head(5), 1):
        print(f"{i}. {headline}")

print("\nProcess Completed Successfully!")