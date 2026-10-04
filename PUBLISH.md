# Publication status

## 当前事实

本会话能读取已连接GitHub仓库，但暴露的GitHub动作没有create repository、write files或push。已尝试发现create/push功能，未取得写入动作。因此：**远端研究仓库尚未创建，文件尚未推送，不能给出一个假装已存在的新仓库链接。**

本包提供统一的本地Git研究仓库；ZIP含文档、配置、代数代码、测试回执和Git元数据。仓库建议名：`a18-physics-native-current-proposal`。发布不是新的物理实验，不能把发布状态当作任何gate通过。

## 在已认证的用户开发环境中发布

先确认已安装Git和GitHub CLI，且`gh auth status`通过。辅助脚本默认private，不会自动公开未发表研究；选择public需明确传参。它不会读取或打印token，不会force-push，不覆盖已有origin，也不上传被.gitignore排除的原始数据。

```sh
python tools/publish_github.py --repo migodam/a18-physics-native-current-proposal --visibility private --confirm
```

需要公开的版本，将visibility明确改为public。执行前应确认owner、研究内容权属、数据范围与导师的公开要求。脚本只推送已提交且clean的working tree；后续实验结果需另做发布审计。

GitHub CLI官方命令说明：<https://cli.github.com/manual/gh_repo_create>（2026-10-04核对）。核心操作为 `gh repo create OWNER/NAME --source=. --remote=origin --push --private`；`--public`替换visibility。

若远端同名repo已存在，脚本返回错误，不覆盖。发布后由真实`git remote -v`及远端commit验证新URL，再更新publication receipt。这里不预填一个成功URL。
