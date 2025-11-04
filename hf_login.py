#!/usr/bin/env python3
"""
Helper script to login to Hugging Face Hub
"""

import os
from huggingface_hub import login

def main():
    print("🔐 Hugging Face Login Helper")
    print("=" * 60)
    
    # Check if token is already set
    token = os.environ.get('HF_TOKEN')
    if token:
        print(f"✅ HF_TOKEN environment variable is already set")
        try:
            login(token=token)
            print("✅ Successfully logged in using HF_TOKEN!")
            return
        except Exception as e:
            print(f"❌ Error logging in with HF_TOKEN: {e}")
            print("   Please check your token is valid")
    
    # Try to login interactively
    print("\nPlease enter your Hugging Face token:")
    print("(Get it from: https://huggingface.co/settings/tokens)")
    print("(Make sure it has 'Write' permissions)")
    
    token = input("\nToken: ").strip()
    
    if not token:
        print("❌ No token provided!")
        return
    
    try:
        login(token=token)
        print("\n✅ Successfully logged in!")
        print("\n💡 Tip: To avoid entering your token each time, you can:")
        print("   - Set environment variable: $env:HF_TOKEN='your_token' (PowerShell)")
        print("   - Or save it in a .env file (not recommended for security)")
    except Exception as e:
        print(f"\n❌ Error logging in: {e}")
        print("   Please check your token is valid and has 'Write' permissions")

if __name__ == "__main__":
    main()

