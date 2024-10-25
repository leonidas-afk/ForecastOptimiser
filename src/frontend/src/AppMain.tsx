import "./App.css";
import { contentStyle, footerStyle, headerStyle, layoutStyle } from "./styles";
import {
  Table,
  Flex,
  Button,
  Form,
  Select,
  Layout,
  Col,
  Row,
  DatePicker,
  Spin,
} from "antd";
import axios, { AxiosResponse } from "axios";
import { flattenDeep, get } from "lodash";
import moment from "moment";
import { PlotRelayoutEvent } from "plotly.js";
import { useState, useEffect } from "react";
import Plot from "react-plotly.js";
import { Navigate } from "react-router-dom";

const { RangePicker } = DatePicker;
const { Header, Footer, Content } = Layout;

const URL = import.meta.env.VITE_APP_LOCALAPI_URL;

function findIndices(
  dataXaxis: string[],
  startTime: string,
  endTime: string
): number[] {
  const start = new Date(startTime).getTime();
  const end = new Date(endTime).getTime();

  const indices: number[] = [];

  for (let i = 0; i < dataXaxis.length; i++) {
    const currentTime = new Date(dataXaxis[i]).getTime();
    if (currentTime >= start && currentTime <= end) {
      indices.push(i);
    }
  }

  return indices;
}

function findExtraPoints(
  dataXAxis: string[],
  endTime: string,
  numberOf4Points: number
) {
  const end = new Date(endTime).getTime();
  const indices: number[] = [];
  for (let i = 0; i < dataXAxis.length; i++) {
    const currentTime = new Date(dataXAxis[i]).getTime();
    if (currentTime > end) {
      indices.push(i);
    }
  }
  return indices.slice(0, numberOf4Points);
}
function AppMain() {
  const [colWidth, setColWidth] = useState(0);
  const [plotData, setPlotData] = useState<any[]>([
    {
      x: [],
      y: [],
      type: "scatter",
      mode: "lines+markers",
      name: "red",
      marker: { color: "red" },
    },
  ]);
  const [backupPlotData, setBackupPlotData] = useState<{
    x: string[];
    y: number[];
  }>({
    x: [],
    y: [],
  });
  const [loading, setLoading] = useState(false);
  const [timeWindow, setTimeWindow] = useState<string>("15m");
  const [forecastHorizon, setForecastHorizon] = useState<string>("1h");
  const [dateRange, setDateRange] = useState<[string, string] | null>(null);
  const [dataTable, setDataTable] = useState({
    fetchedData: [],
    rmse: [],
    algorithm: [],
  });

  useEffect(() => {
    // Function to calculate and set the width of the column
    const updateColWidth = () => {
      const containerWidth = document.querySelector('.ant-row')?.clientWidth || 0;
      const calculatedWidth = (17.2 / 24) * containerWidth;
      setColWidth(calculatedWidth);
    };

    // Update the width initially and on window resize
    updateColWidth();
    window.addEventListener('resize', updateColWidth);

    // Cleanup listener on unmount
    return () => window.removeEventListener('resize', updateColWidth);
  }, []);


  const [form] = Form.useForm();
  // console.log(import.meta.env.VITE_APP_NAME);

  const isLoggedIn = localStorage.getItem("isLoggedIn");

  if (!isLoggedIn) {
    return <Navigate to="/" />;
  }

  const fetchDataFromDb = async () => {
    const values = form.getFieldsValue();
    const { date_range, time_window } = values;
    console.log(date_range);
    const response: AxiosResponse<{
      data: [
        {
          datetime: string;
          value: number;
        }
      ];
    }> = await axios.post(`/localapi/get-data`, {
      table: `LSTM${time_window}`,
      startDate: moment(date_range[0].$d).format("YYYY-MM-DD HH:mm:ss"),
      endDate: moment(date_range[1].$d).format("YYYY-MM-DD HH:mm:ss"),
    });
    console.log(response.data);
    const tmpX = response.data.data.map((d) => d.datetime);
    const tmpY = response.data.data.map((d) => d.value);
    setPlotData([
      {
        x: tmpX,
        y: tmpY,
        type: "scatter",
        mode: "lines+markers",
        name: "red",
        marker: { color: "red" },
      },
    ]);
    setBackupPlotData({
      x: tmpX,
      y: tmpY,
    });
  };

  const fetchData = async () => {
    console.log(plotData[0].y);
    const response = await axios.post(`${URL}?resolution=${timeWindow}&Horizon=${forecastHorizon}`, {
      values: plotData[0].y,
    });
    return response.data;
    // return cleanAndExtractNumbers(response.data);
  };

  function generateXArray(x: string[]): string[] {
    const lastTime = new Date(x[x.length - 1] + "Z"); // Treat input as UTC
    const incrementMinutes = 15; // ToDo: change 15 with a variable function of timeWindow
    const xArray: string[] = [];

    for (let i = 1 + 4; i <= x.length + 4; i++) {
      // ToDo: change 4 with a variable function of timeWindow
      const newTime = new Date(
        lastTime.getTime() + i * incrementMinutes * 60000
      );

      const year = newTime.getUTCFullYear();
      const month = String(newTime.getUTCMonth() + 1).padStart(2, "0");
      const day = String(newTime.getUTCDate()).padStart(2, "0");
      const hours = String(newTime.getUTCHours()).padStart(2, "0");
      const minutes = String(newTime.getUTCMinutes()).padStart(2, "0");
      const seconds = String(newTime.getUTCSeconds()).padStart(2, "0");

      const formattedTime = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
      xArray.push(formattedTime);
    }

    return xArray;
  }

  const fetchPlotData = async () => {
    // xArray should be of length plotData[0].x.length
    // xArray should have values in the form "2023-02-08 00:15:00"
    // xArray should increase the values by 15 minutes taken from the last element of plotData[0].x
    setDataTable({
      fetchedData: [],
      rmse: [],
      algorithm: [],
    });
    const fetchedResponse = await fetchData();
    const fetchedData = flattenDeep(fetchedResponse.predictions);
    setDataTable({
      fetchedData: fetchedData,
      rmse: flattenDeep(fetchedResponse.rmse),
      algorithm: flattenDeep(fetchedResponse.algorithm),
    });
    console.log(plotData, fetchedData);

    return {
      x: generateXArray(plotData[0].x),
      y: fetchedData,
      type: "scatter",
      mode: "lines+markers",
      name: "Predicted values",
      marker: { color: "green" },
    };
  };
  const updatePlotData = async () => {
    setLoading(true); // Start loading
    try {
      const fetchedData = await fetchPlotData();
      console.log("a", fetchedData);
      const indices = findExtraPoints(
        backupPlotData.x,
        plotData[0].x[plotData[0].x.length - 1],
        fetchedData.y.length + 4 //ToDo: change 3 with a variable function of timeWindow + horizon
      );
      const dataXSelected = indices.map((index) => backupPlotData.x[index]);
      const dataYSelected = indices.map((index) => backupPlotData.y[index]);
      

      setPlotData([
        ...plotData.filter((i)=>i.marker.color === "red"),
        {
          x: dataXSelected,
          y: dataYSelected,
          type: "scatter",
          mode: "lines+markers",
          marker: { color: "orange" },
          name: "Actual values",
        },
        fetchedData,
      ]);
    } catch (error) {
      console.error("Error updating plot data:", error);
    } finally {
      setLoading(false); // End loading
    }
  };
  const handleSelection = async (event: Readonly<PlotRelayoutEvent>) => {
    if (event && !("xaxis.autorange" in event)) {
      // console.log(event);
      const indices = findIndices(
        backupPlotData.x,
        (event["xaxis.range[0]"] as number).toString(),
        (event["xaxis.range[1]"] as number).toString()
      );
      // get dataXSelected and dataYSelected based on indices
      const dataXSelected = indices.map((index) => backupPlotData.x[index]);
      const dataYSelected = indices.map((index) => backupPlotData.y[index]);
      setPlotData([
        {
          x: dataXSelected,
          y: dataYSelected,
          type: "scatter",
          mode: "lines+markers",
          name: "Selected actual values",
          marker: { color: "red" },
        },
      ]);
    }

    if ("xaxis.autorange" in event && "yaxis.autorange" in event) {
      console.log("autorange", event);
      setPlotData([
        {
          x: backupPlotData.x,
          y: backupPlotData.y,
          type: "scatter",
          mode: "lines+markers",
          marker: { color: "red" },
        },
      ]);
    }
  };
  const columns = [
    {
      title: "Predicted value",
      dataIndex: "predicted_values",
      key: "predicted_values",
    },
    {
      title: "Algorithm",
      dataIndex: "algorithm",
      key: "algorithm",
    },
    {
      title: "RMSE",
      dataIndex: "rmse",
      key: "rmse",
    },
  ];

  const getTableData = (dataTable) => {
    return dataTable.fetchedData.map((data, index) => {
      return {
        key: index,
        predicted_values: data,
        algorithm: dataTable.algorithm[index],
        rmse: Number.parseFloat(dataTable.rmse[index]).toFixed(5),
      };
    });
  };

  return (
    <Flex gap="middle" wrap>
      <Layout style={layoutStyle}>
        <Header style={headerStyle}>PV Demo App</Header>
        <Content style={contentStyle}>
          <Spin spinning={loading}>
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Form
                  form={form}
                  name="basic"
                  labelCol={{ span: 8 }}
                  wrapperCol={{ span: 16 }}
                  style={{ maxWidth: 600, marginTop: 20 }}
                  initialValues={{
                    time_window: timeWindow,
                  }}
                  onFinish={updatePlotData}
                  onFinishFailed={() => {}}
                  autoComplete="off"
                >
                  <Form.Item label="Resolution" name="time_window">
                    <Select
                      placeholder="Select resolution"
                      onChange={(value) => setTimeWindow(value)}
                      options={[
                        { value: "15m", label: <span>15 minutes</span> },
                        { value: "30m", label: <span>30 minutes</span> },
                        { value: "1h", label: <span>1 hour</span> },
                        { value: "2h", label: <span>2 hours</span> },
                        { value: "4h", label: <span>4 hours</span> },
                        { value: "8h", label: <span>8 hours</span> },
                        { value: "1d", label: <span>1 day</span> },
                        { value: "1w", label: <span>1 week</span> },
                        { value: "1M", label: <span>1 month</span> },
                      ]}
                    />
                  </Form.Item>
                  <Form.Item label="Forecast horizon" name="forecast_horizon">
                    <Select
                      placeholder="Select forecast horizon"
                      onChange={(value) => setForecastHorizon(value)}
                      options={[{ value: "1", label: <span>1</span> },
                        { value: "2", label: <span>2</span> },
                        { value: "3", label: <span>3</span> },
                        { value: "4", label: <span>4</span> },
                        { value: "5", label: <span>5</span> },
                        { value: "6", label: <span>6</span> },
                        { value: "7", label: <span>7</span> }]}
                    />
                  </Form.Item>
                  <Form.Item label="From - To" name="date_range">
                    <RangePicker
                      showTime
                      onChange={(dates, dateStrings) =>
                        setDateRange(dateStrings)
                      }
                      style={{ width: "100%" }}
                    />
                  </Form.Item>
                  <Form.Item>
                    <Button
                      type="primary"
                      onClick={fetchDataFromDb}
                      style={{marginRight: "20px"}}
                    >
                      Load data
                    </Button>
                    <Button
                      type="primary"
                      htmlType="submit"
                    >
                      Get predictions
                    </Button>
                  </Form.Item>
                </Form>
                <Table pageSize={10} dataSource={getTableData(dataTable)} columns={columns} />
              </Col>
              <Col span={18}>
                <Plot
                  data={plotData}
                  onRelayout={(event) => handleSelection(event as never)}
                  layout={{ width: colWidth, height: 660, title: "Data Plot" }}
                />
              </Col>
            </Row>
          </Spin>
        </Content>

        <Footer style={footerStyle}>
          <div>
            <div style={{ float: "left", width: "100px" }}>
              <img src="targetx.png" style={{ width: "100px" }} />
            </div>
            <div style={{ float: "right", width: "100px" }}>
              <img src="lnlogo.png" style={{ width: "100px" }} />
              &copy; Lamda Networks
            </div>
          </div>
        </Footer>
      </Layout>
    </Flex>
  );
}

export default AppMain;
