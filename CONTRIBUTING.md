# 贡献指南

感谢您对向量数据库MCP服务项目的关注！我们欢迎各种形式的贡献，包括但不限于：

- 报告问题
- 提交功能请求
- 提交代码修复或新功能
- 改进文档

## 开发环境设置

1. 克隆仓库：

```bash
git clone https://github.com/your-org/vector-db-service.git
cd vector-db-service
```

2. 创建虚拟环境：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装开发依赖：

```bash
pip install -e ".[dev]"
```

## 代码风格

我们使用以下工具来保持代码质量：

- Black：代码格式化
- isort：导入排序
- mypy：类型检查
- flake8：代码风格检查

在提交代码前，请运行以下命令确保代码符合规范：

```bash
# 格式化代码
black src tests examples
isort src tests examples

# 类型检查
mypy src

# 代码风格检查
flake8 src tests examples
```

## 测试

添加新功能或修复bug时，请同时添加相应的测试。运行测试：

```bash
pytest
```

## 提交PR

1. 创建一个新分支：

```bash
git checkout -b feature/your-feature-name
```

2. 进行修改并提交：

```bash
git add .
git commit -m "描述你的修改"
```

3. 推送到GitHub：

```bash
git push origin feature/your-feature-name
```

4. 创建Pull Request

## MCP服务开发指南

如果您想扩展MCP服务的功能，请参考以下步骤：

1. 在`src/mcp/server.py`中的`MCPServer`类中添加新的工具方法
2. 在`tools`字典中注册新的工具
3. 在`send_server_info`方法中更新工具描述
4. 添加相应的测试用例

## 许可证

通过提交代码，您同意您的贡献将在MIT许可证下发布。