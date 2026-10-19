---
name: aws-profile-switch
description: List the user's configured AWS CLI profiles and switch the active one for this session.
---

# AWS profile switch

Use when the user says "use the staging account" or a command fails with the wrong AWS account.

## Steps

1. List profile names only. This reads profile headers, not keys:
   ```bash
   aws configure list-profiles
   ```
2. Ask the user which profile to use if their request is ambiguous.
3. Set it for the current shell session and confirm which identity is now active:
   ```bash
   export AWS_PROFILE=staging
   aws sts get-caller-identity --query '[Account,Arn]' --output text
   ```
4. Show the account id and ARN so the user can confirm it is the account they expect before any further AWS command.

## Boundaries

- Do not open, print or copy `~/.aws/credentials`, and do not run `aws configure get aws_secret_access_key`.
- If the profile uses SSO and the session expired, tell the user to run `aws sso login --profile staging` themselves; it opens their browser.
