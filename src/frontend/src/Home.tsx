import { Form, Input, Button, Checkbox, message } from "antd";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Home = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const onFinish = (values) => {
    setLoading(true);
    const { username, password } = values;

    if (username === "admin" && password === "targetx") {
      localStorage.setItem("isLoggedIn", true);
      message.success("Login successful!");
      navigate("/main");
    } else {
      message.error("Invalid credentials, please try again.");
    }

    setLoading(false);
  };

  return (
    <div style={{ width: "30%", margin: "auto" }}>
      <h2>PV Prediction Platform Login</h2>
      <Form
        name="login-form"
        onFinish={onFinish}
        layout="vertical"
        initialValues={{ remember: true }}
      >
        <Form.Item
          label="Username"
          name="username"
          rules={[{ required: true, message: "Please input your username!" }]}
        >
          <Input />
        </Form.Item>

        <Form.Item
          label="Password"
          name="password"
          rules={[{ required: true, message: "Please input your password!" }]}
        >
          <Input.Password />
        </Form.Item>

        <Form.Item name="remember" valuePropName="checked">
          <Checkbox>Remember me</Checkbox>
        </Form.Item>

        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading} block>
            Login
          </Button>
        </Form.Item>
      </Form>
      <div style={{ width: "250px", margin: "auto" }}>
        <div style={{ float: "left", width: "100px" }}>
          <img src="targetx.png" style={{ width: "100px" }} />
        </div>
        <div style={{ float: "right", width: "100px" }}>
          <img src="lnlogo.png" style={{ width: "100px" }} />
        </div>
      </div>
    </div>
  );
};

export default Home;
