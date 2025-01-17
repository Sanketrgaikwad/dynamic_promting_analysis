### Example of Dynamic Prompting

Let’s consider a **chatbot** designed to help users with SQL queries. The chatbot uses dynamic prompting to adjust its response based on user input.

---

#### **Scenario 1**: Static Prompting
The prompt is fixed and does not adapt to user input.

**User Input**:  
"Show me total sales."

**Prompt (Static)**:  
"Write an SQL query to display total sales from a database."

**Model Output**:  
```sql
SELECT SUM(sales) AS total_sales FROM sales_table;
```

---

#### **Scenario 2**: Dynamic Prompting
The prompt dynamically adapts to the user’s context and intent.

**User Input 1**:  
"Show me total sales."

**Dynamic Prompt**:  
"The user is asking for the total sales from the sales_table in the database. Generate an SQL query accordingly."  

**Model Output**:  
```sql
SELECT SUM(sales) AS total_sales FROM sales_table;
```

**User Input 2**:  
"Now group it by region."

**Dynamic Prompt**:  
"The user wants to refine the query to group total sales by region. Use the sales_table and include a GROUP BY clause."  

**Model Output**:  
```sql
SELECT region, SUM(sales) AS total_sales 
FROM sales_table 
GROUP BY region;
```

---

### **Real-Life Applications**

#### **1. Adaptive Question Generation**
For a learning assistant:
- **Static Prompt**: "Generate a question about machine learning."
- **Dynamic Prompt**: "The user has been studying supervised learning. Generate a question related to supervised learning."

#### **2. Contextual Recommendations**
For an e-commerce platform:
- **Static Prompt**: "Recommend a product."
- **Dynamic Prompt**: "The user recently viewed electronic gadgets. Recommend products related to electronics."

---

### Benefits of the Dynamic Approach
- **Context-Aware Responses**: Tailors the output to user needs.
- **Interactive Adaptation**: Builds on previous interactions, enhancing continuity.
- **Precision**: Reduces ambiguity by using the user’s specific context.

Would you like an example tailored to one of your projects, such as SQL generation or text similarity?