# 🔬 InsightLab — Firebase Setup Guide (Beginner-Friendly)

Welcome to the **InsightLab Firebase Cloud System**!

InsightLab is built with a **Guest-First philosophy**:
> **Visitors never have to log in or create an account** to upload data, inspect columns, clean missing values, run statistics, create charts, detect anomalies, train machine learning models, generate executive summaries, or download PDF/Excel reports.
>
> An account is **only requested** when a user decides to **permanently save** their projects, datasets, analysis history, or trained models to the cloud.

---

## ⚠️ Important: Separate Project Reminder

> [!IMPORTANT]
> **Do NOT use your existing UCP website Firebase project!**
> Create a **brand new, dedicated Firebase project** specifically for InsightLab (e.g., `insightlab-app`). This guarantees complete isolation between your applications.

---

## Step 1: Create a New Firebase Project (Free Spark Plan)

1. Open the [Firebase Console](https://console.firebase.google.com/) in your web browser.
2. Click **Add project** (or **Create a project**).
3. Project Name: Enter **`insightlab`** (or `insightlab-analytics`).
4. Google Analytics: You can leave this **disabled** (not required) to keep things simple.
5. Click **Create project** and wait a few seconds until your project is ready.
6. Click **Continue**. You are on the **Spark Plan (Free)**.

---

## Step 2: Enable Firebase Authentication (Email & Password)

1. In the left-hand navigation menu of the Firebase Console, click **Build** → **Authentication**.
2. Click the **Get started** button.
3. Under the **Sign-in method** tab, click on **Email/Password**.
4. Toggle the **Enable** switch for **Email/Password** to **ON**.
   *(Leave Email link / passwordless disabled)*.
5. Click **Save**.

That's it! Your users can now register, sign in, and reset their passwords securely.

---

## Step 3: Create Cloud Firestore Database

Cloud Firestore stores user profiles, project snapshots, analysis records, and machine learning configurations.

1. In the left-hand menu, click **Build** → **Firestore Database**.
2. Click **Create database**.
3. **Database ID:** Keep `(default)`.
4. **Location:** Pick a region close to you (e.g., `nam5 (us-central)` or `europe-west`).
5. **Security rules:** Choose **Start in production mode** (we will apply custom rules next).
6. Click **Create**.
7. Once created, click the **Rules** tab at the top of the Firestore screen.
8. Replace the default rules with the contents of [`firestore.rules`](file:///c:/Users/HP/Downloads/insightlab/firestore.rules):

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;

      match /{subcollection}/{documentId} {
        allow read, write: if request.auth != null && request.auth.uid == userId;
      }
    }
    match /{document=**} {
      allow read, write: false;
    }
  }
}
```

9. Click **Publish**. Now each user can strictly access only their own `/users/{uid}/...` documents!

---

## Step 4: Firebase Cloud Storage (Information & Plan Eligibility)

> [!NOTE]
> **Understanding Firebase Storage on the Free (Spark) Plan:**
> - Firebase Cloud Storage allows users to store raw files (CSV, Excel, PDF reports) in the cloud.
> - On newer Google Cloud projects (created after 2024), activating a Cloud Storage bucket may prompt you to link a billing account (Blaze Plan).
> - **InsightLab does NOT require a paid plan.**
> - If you do not configure a Cloud Storage bucket, InsightLab automatically saves all **project snapshots, dataset schemas, analysis discoveries, and ML metrics in Cloud Firestore (100% Free)** and lets users download their files directly to their computer.
> - If you do choose to enable Storage, paste the bucket name in secrets.toml and copy the rules from [`storage.rules`](file:///c:/Users/HP/Downloads/insightlab/storage.rules).

---

## Step 5: Get Your Web API Key & Project ID

1. In the Firebase Console, click the **Gear icon (⚙️)** in the top-left next to *Project Overview* → select **Project settings**.
2. Under the **General** tab:
   - Copy your **Project ID** (e.g., `insightlab-12345`).
   - Copy your **Web API Key** (e.g., `AIzaSy...`).
   - *(Optional)* If you set up Cloud Storage, your bucket name is under the Storage tab (e.g., `insightlab-12345.appspot.com`).

---

## Step 6: Configure `.streamlit/secrets.toml`

Create or edit the file at:
```
c:\Users\HP\Downloads\insightlab\.streamlit\secrets.toml
```

Add your credentials in this format:

```toml
[firebase]
api_key = "AIzaSyYourActualWebApiKeyHere"
project_id = "insightlab-your-actual-project-id"
storage_bucket = "insightlab-your-project-id.appspot.com" # Optional
```

> [!CAUTION]
> Never commit `secrets.toml` to a public Git repository. It is already added to `.gitignore`.

---

## How It Works in InsightLab

1. **Guest Browsing**:
   - Visitors see **👤 Guest Mode** in the sidebar.
   - All 8 analysis studios work instantly with zero prompts.
   
2. **One-Click Sign In**:
   - Clicking **🔑 Sign In / Sign Up** opens a clean sign-in and registration screen.
   - Supports: Name, Email, Password, Password Reset.

3. **Smart Save**:
   - When a guest clicks **☁️ Save to Cloud** in any studio, InsightLab invites them to sign in.
   - Immediately after signing in, InsightLab **automatically completes the pending save** without asking them to re-do their work!

4. **Personal Workspace**:
   - Navigate to **🏠 Personal Workspace** to view:
     - 📂 My Projects
     - 📊 My Datasets
     - 📄 My Reports
     - 📈 My Analysis History
     - 🤖 My Saved Models
     - ⚙️ Account Settings (rename profile, send password reset, sign out)
