#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AlphaFold3 PyTorch 测试运行脚本

此脚本使用模块化框架运行AlphaFold3 PyTorch模型的各种测试。
"""

import sys
import os
from pt_5 import (
    logger, 
    setup_logging, 
    parse_args, 
    Config, 
    TestFramework
)

def main():
    """主函数入口点"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 创建统一配置对象
        config = Config(args)
        
        # 设置日志
        setup_logging(quiet=config.quiet)
        logger.info("AlphaFold3-PyTorch测试脚本启动")
        logger.info(f"配置信息: {config}")
        
        # 创建测试框架并运行所有指定的测试
        framework = TestFramework(config)
        results = framework.run_all_specified_tests()
        
        # 输出测试摘要
        logger.info("测试运行完成")
        for test_name, result in results.items():
            if isinstance(result, bool):
                status = "成功" if result else "失败"
                logger.info(f"测试 {test_name}: {status}")
            elif isinstance(result, dict) and "success" in result:
                status = "成功" if result["success"] else "失败"
                logger.info(f"测试 {test_name}: {status}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断测试运行")
        return 130  # 标准Unix退出码，表示用户中断
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 