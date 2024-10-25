const headerStyle: React.CSSProperties = {
  textAlign: "left",
  color: "#fff",
  height: 64,
  paddingInline: 48,
  lineHeight: "64px",
  backgroundColor: "#4096ff",
};

const contentStyle: React.CSSProperties = {
  textAlign: "center",
  minHeight: "800px",
  lineHeight: "120px",
  color: "#fff",
  backgroundColor: "white",
};

const footerStyle: React.CSSProperties = {
  textAlign: "center",
  color: "#000",
  backgroundColor: "#f7f7f7",
};

const layoutStyle = {
  overflow: "hidden",
  width: "calc(100% - 8px)",
  maxWidth: "calc(100% - 8px)",
};

export { headerStyle, contentStyle, footerStyle, layoutStyle };
