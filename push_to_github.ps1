# GitHub에 빈 저장소 "streamlit_deploy"를 만든 뒤 실행하세요.
# 사용법: .\push_to_github.ps1 -GitHubUsername "본인깃허브아이디"
param(
    [Parameter(Mandatory = $true)]
    [string]$GitHubUsername
)

$env:Path = "C:\Program Files\Git\bin;" + $env:Path
Set-Location $PSScriptRoot

git remote remove origin 2>$null
git remote add origin "https://github.com/$GitHubUsername/streamlit_deploy.git"
git branch -M main

Write-Host "원격: https://github.com/$GitHubUsername/streamlit_deploy.git"
Write-Host "푸시 중... (브라우저 또는 자격 증명 입력이 뜰 수 있습니다)"
git push -u origin main
