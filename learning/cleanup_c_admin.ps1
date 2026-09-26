# ============================================================
#  C 盘深度清理脚本（自动请求管理员权限）
#  运行方式：双击本文件，或在终端里输入
#    powershell -ExecutionPolicy Bypass -File "d:\shuqi\cleanup_c_admin.ps1"
#  弹窗出现时点"是"，即可开始清理。
#
#  清理内容：
#   1. Windows 更新残留缓存（约 13.5G，最大头）
#   2. C:\Windows\Temp 系统临时文件
#   3. 组件清理（WinSxS，DISM，可再释放 1-3G）
#   4. 清空回收站
#  不会删除：个人文档、微信/QQ 聊天记录、HuggingFace 模型、已装软件
# ============================================================

# ---- 自动请求管理员权限（非管理员时重启自身）----
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host '当前不是管理员，正在请求管理员权限，请在弹出的窗口中点"是"...' -ForegroundColor Yellow
    Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`""
    Write-Host '已发起提权请求。若没有弹窗，请手动右键本脚本 -> 以管理员身份运行。'
    exit
}

Write-Host '=== C 盘深度清理开始 ===' -ForegroundColor Cyan
$before = (Get-PSDrive C).Free
Write-Host ('清理前可用空间: ' + [math]::Round($before/1GB,2) + ' GB')

# 1. 停止更新服务，释放对缓存文件的占用
Write-Host '--- 暂停 Windows Update 服务 ---' -ForegroundColor Yellow
Stop-Service wuauserv -Force -ErrorAction SilentlyContinue
Stop-Service bits -Force -ErrorAction SilentlyContinue

# 2. 清理 Windows 更新缓存
Write-Host '--- 清理 SoftwareDistribution\Download ---' -ForegroundColor Yellow
Remove-Item 'C:\Windows\SoftwareDistribution\Download\*' -Recurse -Force -ErrorAction SilentlyContinue
$left = (Get-ChildItem 'C:\Windows\SoftwareDistribution\Download' -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
Write-Host ('更新缓存剩余: ' + [math]::Round($left/1MB,1) + ' MB')

# 3. 系统临时文件
Write-Host '--- 清理 C:\Windows\Temp ---' -ForegroundColor Yellow
Remove-Item 'C:\Windows\Temp\*' -Recurse -Force -ErrorAction SilentlyContinue

# 4. 组件清理（WinSxS，需要几分钟）
Write-Host '--- 运行 DISM 组件清理（可能需要几分钟，请勿关闭窗口）---' -ForegroundColor Yellow
Start-Process -FilePath 'Dism.exe' -ArgumentList '/Online','/Cleanup-Image','/StartComponentCleanup' -NoNewWindow -Wait

# 5. 清空回收站
Write-Host '--- 清空回收站 ---' -ForegroundColor Yellow
Clear-RecycleBin -DriveLetter C -Force -ErrorAction SilentlyContinue

# 恢复更新服务
Start-Service wuauserv -ErrorAction SilentlyContinue
Start-Service bits -ErrorAction SilentlyContinue

$after = (Get-PSDrive C).Free
Write-Host ('清理后可用空间: ' + [math]::Round($after/1GB,2) + ' GB') -ForegroundColor Green
Write-Host ('本次共释放: ' + [math]::Round(($after-$before)/1GB,2) + ' GB') -ForegroundColor Green
Write-Host '=== 清理完成，可关闭窗口 ===' -ForegroundColor Cyan
Read-Host '按回车键退出'
